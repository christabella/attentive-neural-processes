import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import io
import PIL
from torchvision.transforms import ToTensor

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

eps = 1e-5


def plot_rows_df(target_y_rows: pd.DataFrame,
                 context_y_rows: pd.DataFrame,
                 pred_y: np.array,
                 std: np.array,
                 undo_log=False,
                 legend=True):
    if undo_log:
        target_y_rows = np.exp(target_y_rows) - eps
        context_y_rows = np.exp(context_y_rows) - eps
    target_x, target_y = target_y_rows.index, target_y_rows.values
    context_x, context_y = context_y_rows.index, context_y_rows.values
    plot_data(target_x, target_y, context_x, context_y, pred_y, std, legend)


def plot_data(
        target_x,
        target_y,
        context_x,
        context_y,
        pred_y: np.array,
        std: np.array,
        legend=True,
):
    """Plots the predicted mean and variance and the context points.
  
  Args: 
    target_y_rows
    context_y_rows: dataframe with datetime index, and labels
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
    # Plot everything
    j = 0

    # Start with true data and use it to get ylimits (that way they are constant)
    # plt.plot(context_x, context_y, "k:", linewidth=2, label="true")
    # Since C is a subset of T, it's enough to just plot T.
    plt.plot(target_x, target_y, "ko", linewidth=2, label="true", markersize=1)
    ylims = plt.ylim()

    # plot predictions
    plt.plot(target_x, pred_y[0], "b", linewidth=2, label="Predicted mean")
    plt.fill_between(
        target_x,
        pred_y[0, :, 0] - (std[0, :, 0]**2),
        pred_y[0, :, 0] + (std[0, :, 0]**2),
        alpha=0.25,
        facecolor="blue",
        interpolate=True,
        label="Predicted variance",
    )

    # Finally context, we do this with   pandas so it will override x ax and make it nice
    plt.plot(
        context_x,
        context_y,
        "ko",  # black circles
        label="input data",
    )

    # Make the plot pretty
    plt.grid("off")
    plt.ylim(*ylims)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(b=None)
    if legend:
        plt.legend()


def plot_from_loader(loader,
                     model,
                     i=0,
                     undo_log=False,
                     title="",
                     legend=False,
                     context_in_target=None):
    """Plot
    i: Index to visualize
    """
    i = min(int(i), len(loader.dataset) - 1)
    if context_in_target is None:
        context_in_target = model.hparams["context_in_target"]

    device = next(model.parameters()).device
    data = loader.collate_fn([loader.dataset[i]])
    data = [d.to(device) for d in data]
    context_x, context_y, target_x_extra, target_y_extra = data
    target_x = target_x_extra
    target_y = target_y_extra

    model.eval()

    # Get context, like dates, from dataset
    if model.hparams["dataset"] == "smartmeter":
        x_rows, y_rows = loader.dataset.get_rows(i)
        max_num_context = context_x.shape[1]
        y_context_rows = y_rows[:max_num_context]
        y_target_extra_rows = y_rows[max_num_context:]
        dt = y_target_extra_rows.index[0]

        # for the plotting we are doing to run prediction on the context points too
        if not context_in_target:
            target_x = torch.cat([context_x, target_x_extra], 1)
            target_y = torch.cat([context_y, target_y_extra], 1)
        y_target_rows = y_rows

        with torch.no_grad():
            y_pred, kl, loss_test, loss_mse, y_std = model(
                context_x, context_y, target_x, target_y)

            plt.figure()
            plt.title(title + f" loss={loss_test: 2.2g} {dt}")
            plot_rows_df(
                y_target_rows,
                y_context_rows,
                y_pred.detach().cpu().numpy(),
                y_std.detach().cpu().numpy(),
                undo_log=False,
                legend=legend,
            )
    elif model.hparams["dataset"] == "GP":
        with torch.no_grad():
            y_pred, kl, loss_test, loss_mse, y_std = model(
                context_x, context_y, target_x, target_y)

            plt.figure()
            plt.title(title + f" loss={loss_test: 2.2g}")
            plot_data(
                # Flatten the target_x and target_y's shape from [1, N, 1] to just [N]
                target_x.cpu().view(-1),
                target_y.cpu().view(-1),
                context_x.cpu().view(-1),
                context_y.cpu().view(-1),
                y_pred.detach().cpu().numpy(),
                y_std.detach().cpu().numpy(),
                legend=legend,
            )

    return loss_test


def plot_from_loader_to_tensor(*args, **kwargs):
    plot_from_loader(*args, **kwargs)

    # Send fig to tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)  #.unsqueeze(0)
    return image
