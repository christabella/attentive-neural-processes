"""Utilities for creating a PyTorch DataLoader.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from gpflow.kernels import RBF


def generate_GP_data(num_functions=1000, num_samples=100):
    """
    For each function, sample the value at 50 equally spaced
    points in the [−5, 5] interval (Fortuin and Rätsch, 2019).

    Returns:
        Tuple of np.arrays of size (num_samples, 1) and
        (num_samples, num_functions).
    """
    jitter = 1e-6
    Xs = np.linspace(-5.0, 5.0, num_samples)[:, None]
    kernel = RBF(lengthscales=1.)
    cov = kernel(Xs)
    L = np.linalg.cholesky(cov + np.eye(num_samples) * jitter)
    epsilon = np.random.randn(num_samples, num_functions)
    F = np.sin(Xs) + np.matmul(L, epsilon)
    return Xs, F


def collate_fns_GP(max_num_context, context_in_target=True):
    """When automatic batching is enabled (which it is since we passed batch_size), collate_fn is called with a list of data samples at each time. It is expected to collate the input samples into a batch for yielding from the data loader iterator.
    """
    def collate_fn(batch):
        # Collate
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(4, max_num_context)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Sample from numpy arrays along 2nd dim.
        # Similar to https://github.com/EmilienDupont/neural-processes/blob/master/utils.py#L5
        inds = np.random.choice(range(x.shape[1]),
                                size=num_context,
                                replace=False)
        inds.sort()
        x_context, y_context = x[:, inds], y[:, inds]
        x_target, y_target = x, y  # num_extra_target = num_samples - num_context

        return x_context, y_context, x_target, y_target

    return collate_fn


class GPCurvesDataset(Dataset):
    def __init__(self, X, F):
        """Accepts tuple of np.arrays of size (num_samples, 1) and
        (num_samples, num_functions).
        """
        self.X, self.F = X, F

    def __getitem__(self, i):
        # Shape (num_samples, x_dim)
        X = self.X  # Always the same for now... TODO: Maybe should vary!
        # Shape (num_samples, y_dim)
        Y = self.F[:, i][:, None]
        # TODO should the context-target splits be the same, or different, for a new iteration?
        return torch.from_numpy(X), torch.from_numpy(Y)

    def __len__(self):
        return self.F.shape[1]
