# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys

import numpy as np
import torch

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def autoregressive(X, w):
    return np.array(
        [np.sum(X[0:k + 1] * np.flip(w[0:k + 1]).reshape(-1, 1)) for k in
         range(len(X))])


def create_autoregressive_data(n_samples=100,
                               seq_len=6,
                               n_features=1,
                               X_m=1,
                               X_v=2,
                               noise_profile=None,
                               memory_factor=0.9,
                               mode="time-dependent"):
    # TODO varying series lengths (perhaps with generated padding)
    # TODO sampling frequencies
    # TODO stationarity
    # TODO automatic noise profile

    # Create the input features
    X = [np.random.normal(X_m, X_v, (seq_len, n_features)) for _ in
         range(n_samples)]
    w = np.array([memory_factor ** k for k in range(seq_len)])

    if noise_profile is None:
        # default increasing noise profile
        noise_profile = np.array(
            [1 / (seq_len - 1) * k for k in range(seq_len)])

    assert len(noise_profile) == seq_len

    Y = None  # Y stores the actual time series values generated from features X
    if mode == "noise-sweep":
        Y = [[(autoregressive(X[k], w).reshape(seq_len, n_features) +
               np.random.normal(0, noise_profile[u],
                                (seq_len, n_features))).reshape(seq_len, )
              for k in range(n_samples)] for u in range(len(noise_profile))]

    elif mode == "time-dependent":
        Y = [(autoregressive(X[k], w).reshape(seq_len, n_features) + (
            torch.normal(mean=0.0, std=torch.tensor(noise_profile)))
              .detach().numpy().reshape(-1, n_features)).reshape(seq_len, )
             for k in range(n_samples)]

    return X, Y


def generate_autoregressive_forecast_dataset(n_samples=100,
                                             seq_len=100,
                                             n_features=1,
                                             X_mean=1,
                                             X_variance=2,
                                             noise_profile=None,
                                             memory_factor=0.9,
                                             mode="time-dependent",
                                             horizon=10):
    total_seq_len = seq_len + horizon
    # Create the input features of the generating process
    X_gen = [np.random.normal(X_mean, X_variance, (total_seq_len,
                                                   n_features))
             for _ in range(n_samples)]
    w = np.array([memory_factor ** k for k in range(total_seq_len)])

    if noise_profile is None:
        # default increasing noise profile
        noise_profile = np.array(
            [1 / (seq_len - 1) * k for k in range(total_seq_len)])

    X = None  # X stores the time series values generated from features X_gen
    if mode == "noise-sweep":
        X = torch.FloatTensor(
            [[(autoregressive(X_gen[k], w).reshape(total_seq_len, n_features) +
               np.random.normal(0, noise_profile[u], (total_seq_len,
                                                      n_features)))
                  .reshape(total_seq_len, ) for k in range(n_samples)]
             for u in range(len(noise_profile))])


    elif mode == "time-dependent":
        X = torch.FloatTensor(
            [(autoregressive(X_gen[k], w)
              .reshape(total_seq_len, n_features) + (
                  torch.normal(mean=0.0, std=torch.tensor(noise_profile)))
              .detach().numpy().reshape(-1, n_features)).reshape(
                total_seq_len, )
                for k in range(n_samples)])

    Y = torch.FloatTensor(X[:, -horizon:])  # `horizon` of predictions
    X = torch.nn.utils.rnn.pad_sequence(X[:, -horizon], batch_first=True)

    dataset = torch.utils.data.TensorDataset(X, Y)
    return dataset
