from argparse import ArgumentParser
import io

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
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
#
# For an in-depth discussion on this topic, see *(Fortuin and Rätsch, 2019)*.
# This notebook reproduces section 4.2 of this paper.

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %%
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


# %%
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


# %%
def generate_meta_and_test_tasks(num_context, num_meta, num_test, N):
    """Generates meta-task datasets {D_i} = {x_i, y_i} and target task training
    and test data {\tilde{x}, \tilde{y}} (Fortuin and Rätsch, 2019).

    Args:
        num_context: The number of training points, \tilde{n} in Table S1.
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
    Xs, F = generate_GP_data(num_functions=num_meta + num_test, N=N)
    meta = []
    # sd = 1e-1  # Standard deviation for normal observation noise.
    for i in range(num_meta):
        # We always use all data points of the curve to train mean function,
        # i.e. n_i = N.
        # noise = sd * np.random.randn(N, 1)
        Y = F[:, i][:, None]
        meta.append((Xs, Y))
    test = []
    for i in range(num_test):
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
        test.append(((x_context, y_context), (x_target, y_target)))
    return meta, test


# ## Create the mean function
# We will use a Keras model Deep Neural Network as mean function.
def build_mean_function():
    inputs = tf.keras.layers.Input(shape=(1, ))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
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


def run_adam(model, iterations, lr):
    """
    Utility function running the Adam optimizer.
    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    adam = AdamW(learning_rate=lr, weight_decay=0.01)
    optimization_step = create_optimization_step(adam, model)
    LML = 0
    for step in range(iterations):
        loss = optimization_step()  # This is the LML
        LML += loss.numpy()
    LML /= iterations

    return LML


# Next, we define the training loop for metalearning.


def train_loop(meta_tasks, num_epochs, num_iters, lr):
    """
    Metalearning training loop. Trained for 100 epochs in original experiment.
    :param meta_tasks: list of metatasks.
    :param num_epochs: number of iterations of tasks set
    :returns: a mean function object
    """
    # Initialize mean function
    mean_function = build_mean_function()
    # Iterate for several passes over the tasks set
    for iteration in range(num_epochs):
        ts = time.time()
        print("Currently in meta-iteration/epoch {}".format(iteration))
        # Iterate over tasks
        for i, task in enumerate(meta_tasks):
            data = task  # (X, Y)
            model = build_model(data, mean_function=mean_function)
            train_loss = run_adam(model, num_iters, lr)
            tf.summary.scalar(
                'train_loss',
                train_loss,
                # Each step corresponds to a run_adam over one task
                step=iteration * len(meta_tasks) + i)

        print(">>>> Epoch took {:.2f} s".format(time.time() - ts))

    return mean_function


# %%
def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred)**2)


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
    meta, test = generate_meta_and_test_tasks(hparams.num_context,
                                              hparams.num_tasks_train,
                                              hparams.num_tasks_test,
                                              hparams.num_samples)

    mean_function_optimal = train_loop(meta,
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
    mean_squared_errors = []
    for i, test_task in enumerate(test):
        figure = plt.figure(figsize=(8, 4))
        (train_X, train_Y), (Xs, F) = test_task  # train_X and Xs are disjoint
        pred_mean, pred_var = test_models[i].predict_f(Xs)
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
        mse = mean_squared_error(F, pred_mean)
        mean_squared_errors.append(mse)
        plt.title(f"Test Task {i + 1} | MSE = {mse:.2f}")
        plt.legend()
        # Send fig to tensorboard
        tf.summary.image("test_image", plot_to_image(figure), step=i)
        # plt.show()

    # %%
    mean_mse = np.mean(mean_squared_errors)
    std_mse = np.std(mean_squared_errors) / np.sqrt(hparams.num_tasks_test)
    print(f"The mean MSE over all {hparams.num_tasks_test} test tasks"
          f"is {mean_mse:.2f} +/- {std_mse:.2f}")
    tf.summary.scalar("test_mse_functional", mean_mse)


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
