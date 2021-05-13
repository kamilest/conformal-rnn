# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import torch


def autoregressive(X_gen, w):
    """ Generates a single time series example. """
    return np.array(
        [np.sum(X_gen[0:k + 1] * np.flip(w[0:k + 1]).reshape(-1, 1)) for k in
         range(len(X_gen))])


def create_autoregressive_data(n_samples=100,
                               seq_len=6,
                               n_features=1,
                               X_m=1,
                               X_v=2,
                               noise_profile=None,
                               memory_factor=0.9,
                               mode="time-dependent"):
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
                                             noise_profiles=None,
                                             memory_factor=0.9,
                                             mode="time-dependent",
                                             horizon=10):

    # TODO replace total_seq_len with sampled sequence lengths.
    sequence_lengths = [seq_len + horizon] * n_samples
    max_seq_len = np.max(sequence_lengths)

    # Examples with sequence lenghts <=`horizon` don't give any
    # information and should be excluded.
    # assert np.min(sequence_lengths) > horizon
    assert sequence_lengths == n_samples

    # Create the input features of the generating process
    X_gen = [np.random.normal(X_mean, X_variance, (seq_len,
                                                   n_features))
             for seq_len in sequence_lengths]

    w = np.array([memory_factor ** k for k in range(max_seq_len)])

    if noise_profiles is None:
        # Default increasing noise profile.
        # TODO sampling frequencies
        # TODO stationarity
        noise_profiles = [
            [1 / (seq_len - 1) * k for k in range(max(1, seq_len))] for seq_len
            in sequence_lengths]

    # if mode == "noise-sweep":
    #     X = [[(autoregressive(X_gen[k], w).reshape(sequence_lengths[k],
    #                                                n_features) +
    #            np.random.normal(0, noise_profile[u], (sequence_lengths[k],
    #                                                   n_features)))
    #               .reshape(sequence_lengths[k], ) for k in range(n_samples)]
    #          for u in range(len(noise_profile))]

    # X_full stores the time series values generated from features X_gen.
    X_full = []
    if mode == "time-dependent":
        ar = [autoregressive(x, w).reshape(-1, n_features) for x in X_gen]
        noise = [np.random.normal(0., noise_profile).reshape(-1, n_features) for
                 noise_profile in noise_profiles]
        X_full = [i + j for i, j in zip(ar, noise)]

    # Splitting time series into training sequence X and target sequence Y;
    # Y stores the time series predicted targets `horizon` steps away
    X, Y = [], []
    for seq in X_full:
        seq_len = len(seq)
        if seq_len >= 2 * horizon:
            X.append(seq[:-horizon])
            Y.append(seq[-horizon:])
        elif seq_len > horizon:
            X.append(seq[:seq_len - horizon])
            Y.append(seq[-(seq_len - horizon):])

    # TODO clean up (un)squeezing.
    # TODO generate padded tensors and the LongTensor of sequence lengths.

    # X: [n_samples, max_seq_len, n_features]
    X_tensor = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

    # Y: [n_samples, horizon, n_features]
    Y_tensor = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)

    return torch.utils.data.TensorDataset(X_tensor, Y_tensor)
