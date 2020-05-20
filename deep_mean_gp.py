from argparse import ArgumentParser
import io
# # Custom mean functions: metalearning with GPs
# One of the advantages of Gaussian process is their flexibility as a modeling
# tool. For instance, if the modeler knows that there is an underlying trend in
# the data, they can specify a mean function that captures this trend.
#
# In this notebook, we illustrate how to use GPflow to construct a custom
# neural network mean function for GPs that can capture complex trends. We look
# at this functionality in the context of metalearning, where a number of
# metatasks are available at train time and the user wants to adapt a flexible
# model to new tasks at test time.
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import time
import gpflow
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPR
from gpflow.base import Parameter
from gpflow import set_trainable
from gpflow.config import default_float
from tensorflow.python.keras import backend as K
from tensorflow_addons.optimizers import AdamW

from src.utils import plot_to_image

K.set_floatx("float64")
assert default_float() == np.float64

# Could have also used this official tutorial for TB monitoring :facepalm:
# https://gpflow.readthedocs.io/en/master/notebooks/basics/monitoring.html
summary_writer = tf.summary.create_file_writer("tensorboard_logs")
summary_writer.set_as_default()

# https://qiita.com/jack_ama/items/491e073cadfdd738bf6c
global_step = np.array(1, dtype=np.int64)
tf.summary.experimental.set_step(global_step)


def generate_GP_data(num_functions=10, N=500):
    """
    For each function, sample the value at `N` equally spaced
    points in the [−5, 5] interval (Fortuin and Rätsch, 2019).

    Returns:
        Tuple of np.arrays of size (N, 1) and (N, num_functions).
    """
    jitter = 1e-6
    Xs = np.linspace(-5.0, 5.0, N)[:, None]
    kernel = RBF(lengthscales=1.0)
    cov = kernel(Xs)
    L = np.linalg.cholesky(cov + np.eye(N) * jitter)
    epsilon = np.random.randn(N, num_functions)
    F = np.sin(Xs) + np.matmul(L, epsilon)
    return Xs, F


def generate_meta_and_test_tasks(num_context, num_meta, num_test, N):
    """Generates meta-task datasets {D_i} = {x_i, y_i} and target task training
    and test data {\tilde{x}, \tilde{y}} (Fortuin and Rätsch, 2019).

    Args:
        num_context: The number of training points, \tilde{n} > N in Table S1.
        num_meta: The number of meta-tasks ("sampled functions").
        num_test: The number of test tasks ("unseen functions").
        N: Number of sampled data points per function.

    Returns:
        A tuple (meta, test) where
        - meta: List of num_meta pairs of arrays (X, Y) of size (n, 1) each.
        - test: List of num_test pairs of pairs of arrays of sizes
                (((num_context, 1), (num_context, 1)),
                 ((N - num_context, 1), (N - num_context, 1))).
    """
    # For splitting into train, val and test.
    Xs, F = generate_GP_data(num_functions=num_meta + 2 * num_test, N=N)
    # Validation needs to be the same format as test, i.e. (x_C, y_C, x_T, y_T)
    meta = []
    # sd = 1e-1  # Standard deviation for normal observation noise.
    for i in range(num_meta):
        # We always use all data points of the curve to train mean function,
        # i.e. n_i = N.
        # noise = sd * np.random.randn(N, 1)
        Y = F[:, i][:, None]
        meta.append((Xs, Y))
    valid_and_test = []
    for i in range(2 * num_test):
        inds = np.random.choice(range(N), size=num_context, replace=False)
        inds.sort()
        # Form target training set, \tilde{D}_{train} (see Figure 1).
        x_context, y_context = Xs[inds], F[inds, num_meta + i][:, None]
        # Form target test set, \tilde{D}_{test}, which is disjoint from the
        # target traininig set as it is in the original implementation.
        x_target, y_target = Xs[:], F[:, num_meta + i][:, None]
        # num_extra_target = num_samples - num_context
        # Form target tasks' datasets \tilde{D} as a pair of pairs,
        # (\tilde{X}, \tilde{Y}) and (\tilde{X}*, \tilde{Y}*).
        valid_and_test.append(((x_context, y_context), (x_target, y_target)))
    return meta, valid_and_test


# ## Create the mean function
# We will use a Keras model Deep Neural Network as mean function.
def build_mean_function():
    inputs = tf.keras.layers.Input(shape=(1, ))
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# %% [markdown]
# ## Build the GP metamodel
# Metalearning boils down to learning a good prior that can generalize
# to new tasks with a small number of data points. This framework is
# prevalent in GP modeling, where we usually maximize the marginal
# likelihood to learn a good set of hyperparameters that specify the
# GP prior.
#
# We perform the same optimization here, while sharing the
# hyperparameters across all the metatasks. For simplicity, we fix the
# kernel and likelihood parameters and learn those only for the mean
# function. Hence, our metalearning procedure is to cycle through the
# metatasks continuously, optimizing their marginal likelihood until a
# convergence criteria is reached (here, we just implement a fixed
# number of iterations over the tasks).
#
# To begin this process, first we create a utility function that takes
# in a task (X, Y) and a mean function and outputs a GP model.

# %%


def build_model(data, mean_function):
    model = GPR(data, kernel=RBF(), mean_function=mean_function)
    set_trainable(model.kernel, False)
    model.likelihood.variance.assign(1e-2)
    set_trainable(model.likelihood, False)
    return model


# %%
def create_optimization_step(optimizer, model: gpflow.models.GPR):
    @tf.function
    def optimization_step():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            objective = -model.log_marginal_likelihood()  # LML
            grads = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return objective

    return optimization_step


def run_adam(model, iterations, data, lr):
    """
    Utility function running the Adam optimizer.
    :param model: GPflow model
    :param interations: number of iterations
    """
    global global_step
    # Create an Adam Optimizer action
    adam = AdamW(learning_rate=lr, weight_decay=0.01)
    optimization_step = create_optimization_step(adam, model)
    for step in range(iterations):
        loss = optimization_step()  # This is the neg LML, a scalar
        tf.summary.scalar('train_loss', loss)
        # This is kinda meaningless though?
        log_density = model.predict_log_density(
            data)  # Vector for each point...
        tf.summary.scalar('train_log_pred_likelihood', np.mean(log_density))
        global_step += 1


# Next, we define the training loop for metalearning.


def train_loop(meta_tasks, valid_tasks, num_epochs, num_iters, lr):
    """
    Metalearning training loop. Trained for 100 epochs in original experiment.
    :param meta_tasks: list of metatasks.
    :param num_epochs: number of iterations of tasks set
    :returns: a mean function object
    """
    global global_step
    # Initialize mean function
    mean_function = build_mean_function()
    # Iterate for several passes over the tasks set
    for iteration in range(num_epochs):
        ts = time.time()
        print("Currently in meta-iteration/epoch {}".format(iteration))
        # Iterate over tasks
        for i, task in enumerate(meta_tasks):
            data = task  # (X, Y)
            # Basically we are discarding S_C. todo(bella) make sure this is correct
            # also in pacoh; well in abstract.py eval() it's passed into predict()
            model = build_model(data, mean_function=mean_function)
            run_adam(model, num_iters, data, lr)
            # Compute validation metrics in a fashion similar to GPR_meta_mll.py's
            # `gp_model.meta_fit(valid_tuples=meta_test_data, log_period=1000, n_iter=20000)`
            # OH in GPR_meta_mll:main, they use the same data for both validation and test...

            (val_X_C, val_Y_C), (val_X_T, val_Y_T) = valid_tasks[i]
            # I need to first fit the GPR model on the val data... silly!
            val_model = build_model((val_X_C, val_Y_C),
                                    mean_function=mean_function)
            pred_mean, pred_var = val_model.predict_y(val_X_T)
            # Convert eager TF tensors to numpy
            pred_mean, pred_var = pred_mean.numpy(), pred_var.numpy()
            calib_error = _calib_error(pred_mean, pred_var**0.5, val_Y_T)
            tf.summary.scalar("valid_calib_error", calib_error)
            mse = mean_squared_error(val_Y_T, pred_mean)
            tf.summary.scalar("valid_mse", mse)
            valid_log_density = val_model.predict_log_density(
                (val_X_T, val_Y_T))
            tf.summary.scalar('valid_log_pred_likelihood',
                              np.mean(valid_log_density))

            # Each step corresponds to a run_adam over one task
            # step=iteration * len(meta_tasks) + i)

        print(">>>> Epoch took {:.2f} s".format(time.time() - ts))

    return mean_function


def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred)**2)


def _calib_error(pred_mean, pred_std, test_y):
    """Pred mean of size (samples, y_dim) and test_y (truth)"""
    pred_dist_vectorized = torch.distributions.normal.Normal(
        torch.tensor(pred_mean), torch.tensor(pred_std))
    test_t_tensor = torch.tensor(test_y)
    cdf_vals = pred_dist_vectorized.cdf(test_t_tensor)

    if test_t_tensor.shape[0] == 1:
        test_t_tensor = test_t_tensor.flatten()
        cdf_vals = cdf_vals.flatten()

    num_points = test_t_tensor.shape[0]
    conf_levels = torch.linspace(0.05, 0.95, 20)
    emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels,
                                        dim=0).float() / num_points

    calib_rmse = torch.sqrt(
        torch.mean((emp_freq_per_conf_level - conf_levels)**2))
    return calib_rmse


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# **NOTE:** We use only 50 metatasks and 10 test tasks over 5 epochs
# for scalability, whereas the paper uses 1,000 and 200 respectively
# over 100 epochs. To compensate, we sample 500 points per curve,
# whereas the paper samples only 50 points. Hence, there might be some
# discrepancies in the results.


def main(hparams):
    # for reproducibility of this notebook:
    np.random.seed(hparams.seed)
    tf.random.set_seed(hparams.seed)
    # Generate the tasks from a GP with an SE kernel and a sinusoidal mean.
    # Each task is a realization of this process.
    meta, valid_and_test = generate_meta_and_test_tasks(
        hparams.num_context, hparams.num_tasks_train, hparams.num_tasks_test,
        hparams.num_samples)
    # Further split
    valid, test = valid_and_test[:hparams.num_tasks_test], valid_and_test[
        hparams.num_tasks_test:]
    mean_function_optimal = train_loop(meta,
                                       valid,
                                       num_epochs=hparams.epochs,
                                       num_iters=hparams.num_iters,
                                       lr=hparams.learning_rate)
    # Finally, we use the optimized mean function for all of the test tasks.
    # **NOTE:** We do not do any further optimization for the hyperparameters.
    test_models = [
        build_model(data, mean_function_optimal) for (data, _) in test
    ]
    # Assess the model
    # We assess the performance of this procedure on the test tasks. For this,
    # we use the mean squared error as a performance metric.
    mean_squared_errors, log_densities, calib_errors, LMLs = [], [], [], []
    # global_step = np.array(1, dtype=np.int64)  # Reset global step...
    for i, test_task in enumerate(test):
        # Full
        m = test_models[i]
        (train_X, train_Y), (Xs, F) = test_task
        pred_mean, pred_var = test_models[i].predict_y(Xs)
        # Convert eager TF tensors to numpy
        pred_mean, pred_var = pred_mean.numpy(), pred_var.numpy()
        #     pred_mean_y, pred_var_y = test_models[i].predict_y(Xs)
        mse = mean_squared_error(F, pred_mean)
        mean_squared_errors.append(mse)
        log_density = test_models[i].predict_log_density(
            (Xs, F))  # Log density for each new X point (45)
        log_densities.append(np.mean(log_density))
        calib_error = _calib_error(pred_mean, pred_var**0.5, F)
        calib_errors.append(calib_error)
        LML = -test_models[i].log_marginal_likelihood()  # LML
        LMLs.append(LML)
        tf.summary.scalar("test_mse",
                          running_mean(mean_squared_errors, i + 1).item(),
                          step=i)
        tf.summary.scalar("test_log_likelihood",
                          running_mean(log_densities, i + 1).item(),
                          step=i)
        tf.summary.scalar("test_calib_error",
                          running_mean(calib_errors, i + 1).item(),
                          step=i)
        tf.summary.scalar("test_LML", running_mean(LMLs, i + 1).item(), step=i)

        figure = plt.figure(figsize=(8, 4))
        plt.plot(Xs,
                 pred_mean,
                 label="Prediction mean",
                 color="blue",
                 linewidth=2)
        plt.fill_between(
            Xs.squeeze(1),
            tf.squeeze(pred_mean - pred_var),
            tf.squeeze(pred_mean + pred_var),
            alpha=0.25,
            facecolor="blue",
            label="Prediction variance",
        )
        plt.plot(train_X, train_Y, "ko", label="Training points")
        plt.plot(Xs, F, "ko", label="Ground truth", linewidth=2, markersize=1)
        # Get log likelihood, MSE, and calibration error a la
        # https://github.com/jonasrothfuss/meta_learning_pacoh/blob/376349e66bdd782e3d06b4bac2ecb56a2a10bcf6/meta_learn/abstract.py#L41
        mse = mean_squared_error(F, pred_mean)
        mean_squared_errors.append(mse)
        # Performs posterior inference (target training) with (context_x,
        # context_y) as training data and then computes the predictive
        # distribution of the targets p(y|test_x, test_context_x, context_y) in
        # the test points
        plt.title(
            f"Task {i} | Log-likelihood={np.mean(log_density): 2.2g}, MSE={mse: 2.2g}, CE={calib_error: 2.2g}"
        )
        # plt.legend()
        # Send fig to tensorboard
        tf.summary.image("test_image", plot_to_image(figure), step=i)
        plt.close()
        # plt.show()
    num_tasks_test = hparams.num_tasks_test
    mean_mse = np.mean(mean_squared_errors)
    std_mse = np.std(mean_squared_errors) / np.sqrt(
        num_tasks_test)  # SD = SE * sqrt(N)
    avg_log_likelihood = np.mean(log_densities)
    std_log_likelihood = np.std(log_densities) / np.sqrt(num_tasks_test)
    calib_error = np.mean(calib_errors)
    std_calib_error = np.std(calib_errors) / np.sqrt(num_tasks_test)
    LML = np.mean(LMLs)
    std_LML = np.std(LMLs) / np.sqrt(num_tasks_test)
    tf.summary.scalar("test_mse", mean_mse, step=1)
    tf.summary.scalar("test_log_likelihood", avg_log_likelihood, step=1)
    tf.summary.scalar("test_calib_error", calib_error, step=1)
    tf.summary.scalar("test_LML", LML, step=1)
    print(
        f"The mean MSE over all {num_tasks_test} test tasks is {mean_mse:.3f} +/- {std_mse:.3f}"
    )
    print(
        f"The avg log likelihood over all {num_tasks_test} test tasks is {avg_log_likelihood:.3f} +/- {std_log_likelihood:.3f}"
    )
    print(
        f"The avg calib_error over all {num_tasks_test} test tasks is {calib_error:.3f} +/- {std_calib_error:.3f}"
    )
    print(
        f"The avg LML over all {num_tasks_test} test tasks is {LML:.3f} +/- {std_LML:.3f}"
    )


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2334)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument(
        '--num_iters',
        type=int,
        default=1,
        help="Inner loop gradient steps. 1 to be comparable with NPs.")

    parser.add_argument('--num_tasks_train', type=int, default=500)
    parser.add_argument('--num_tasks_test', type=int, default=500)
    parser.add_argument('--num_samples', type=int, default=50)
    # TODO: This should vary among functions/meta-datasets?
    parser.add_argument('--num_context', type=int, default=5, help='')
    parser.add_argument('--meta_parameters',
                        type=str,
                        default="mean",
                        help="Which parameters to train over meta-tasks?",
                        choices=["mean", "kernel", "both"])
    parser.add_argument('--dataset',
                        type=str,
                        default='GP',
                        choices=['GP', 'smartmeter'])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='')

    # each LightningModule defines arguments relevant to it
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
