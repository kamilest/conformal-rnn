# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pickle

import numpy as np
import torch


def autoregressive(X_gen, w):
    """ Generates the autoregressive component of a single time series
    example. """
    return np.array(
        [np.sum(X_gen[0:k + 1] * np.flip(w[0:k + 1]).reshape(-1, 1)) for k in
         range(len(X_gen))])


# https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_seasonal.html
def seasonal(duration, periodicity, amplitude=1., harmonics=1):
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = amplitude * np.random.randn(harmonics)
    gamma_star_jt = amplitude * np.random.randn(harmonics)

    total_timesteps = 2 * duration  # Pad for burn in
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (gamma_jt[j - 1] * cos_j
                                 + gamma_star_jt[j - 1] * sin_j
                                 + amplitude * np.random.randn())
            gamma_star_jtp1[j - 1] = (- gamma_jt[j - 1] * sin_j
                                      + gamma_star_jt[j - 1] * cos_j
                                      + amplitude * np.random.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1

    return series[-duration:].reshape(-1, 1)  # Discard burn in


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


class AutoregressiveForecastDataset(torch.utils.data.Dataset):
    """Synthetic autoregressive forecast dataset."""

    def __init__(self, X, Y, sequence_lengths):
        super(AutoregressiveForecastDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]


def generate_autoregressive_forecast_dataset(n_samples=100,
                                             seq_len=100,
                                             n_features=1,
                                             X_mean=1,
                                             X_variance=2,
                                             memory_factor=0.9,
                                             noise_mode='time-dependent',
                                             noise_profile=None,
                                             periodicity=None,
                                             amplitude=1,
                                             harmonics=1,
                                             dynamic_sequence_lengths=False,
                                             horizon=10):
    seq_len = max(seq_len, horizon)

    if noise_profile is None:
        noise_profile = [0.2, 0.4, 0.6, 0.8, 1.]

    if dynamic_sequence_lengths:
        sequence_lengths = horizon + seq_len // 2 \
                           + np.random.geometric(p=2 / seq_len, size=n_samples)
    else:
        sequence_lengths = np.array([seq_len + horizon] * n_samples)

    # Create the input features of the generating process
    X_gen = [np.random.normal(X_mean, X_variance, (seq_len,
                                                   n_features))
             for seq_len in sequence_lengths]

    w = np.array([memory_factor ** k for k in range(np.max(sequence_lengths))])

    if noise_mode == 'time-dependent':
        # Default increasing noise profile.
        # TODO stationarity
        noise_vars = [[noise_profile[(s * len(noise_profile)) // sl]
                       for s in range(sl)] for sl in sequence_lengths]
    elif noise_mode == 'noise-sweep':
        noise_vars = [
            [noise_profile[(s * len(noise_profile)) // len(sequence_lengths)]] *
            sequence_lengths[s] for s in range(len(sequence_lengths))]
    else:
        # No additional noise beyond the variance of X_gen
        noise_vars = [[0] * sl for sl in sequence_lengths]

    # X_full stores the time series values generated from features X_gen.
    ar = [autoregressive(x, w).reshape(-1, n_features) for x in X_gen]
    noise = [np.random.normal(0., nv).reshape(-1, n_features) for
             nv in noise_vars]

    if periodicity is not None:
        periodic = [seasonal(sl, periodicity, amplitude, harmonics) for sl in
                    sequence_lengths]
    else:
        periodic = np.array([np.zeros(sl) for sl in sequence_lengths]) \
            .reshape(-1, 1)

    X_full = [torch.tensor(i + j + k) for i, j, k in zip(ar, noise, periodic)]

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

        # Examples with sequence lenghts <=`horizon` don't give any
        # information and are excluded.
        # assert np.min(sequence_lengths) > horizon

    return X, Y, sequence_lengths


def get_synthetic_splits(length=10, horizon=5, conformal=True,
                         n_train=1000, n_calibration=1000, n_test=500,
                         cached=True,
                         mean=1,
                         variance=2,
                         memory_factor=0.9,
                         noise_mode='time-dependent'):
    # Time series parameters
    periodicity = None
    amplitude = 1
    dynamic_sequence_lengths = False

    if cached:
        datasets = []
        for i in range(1, 6):
            if conformal:
                with open('processed_data/synthetic_{}_conformal_{}.pkl'.format(
                        noise_mode, i),
                        'rb') as f:
                    train_dataset, calibration_dataset, test_dataset = \
                        pickle.load(f)
            else:
                with open('processed_data/synthetic_{}_raw_{}.pkl'.format(
                        noise_mode, i),
                          'rb') as f:
                    train_dataset, calibration_dataset, test_dataset = \
                        pickle.load(f)
            datasets.append((train_dataset, calibration_dataset, test_dataset))
    else:
        datasets = []
        for i in ([2, 10] if noise_mode == 'periodic' else range(1, 6)):
            if noise_mode == 'time-dependent':
                noise_profile = [0.1 * i * k for k in range(length + horizon)]
            else:
                noise_profile = [0.1 * i for _ in range(length + horizon)]

            if noise_mode == 'periodic':
                periodicity = i
                amplitude = 5

            X_train, Y_train, sequence_lengths_train = \
                generate_autoregressive_forecast_dataset(
                    n_samples=n_train,
                    seq_len=length,
                    horizon=horizon,
                    periodicity=periodicity,
                    amplitude=amplitude,
                    X_mean=mean,
                    X_variance=variance,
                    memory_factor=memory_factor,
                    noise_mode=noise_mode,
                    noise_profile=noise_profile,
                    dynamic_sequence_lengths=dynamic_sequence_lengths)

            # X: [n_samples, max_seq_len, n_features]
            X_train_tensor = torch.nn.utils.rnn.pad_sequence(X_train,
                                                             batch_first=True).float()

            # Y: [n_samples, horizon, n_features]
            Y_train_tensor = torch.nn.utils.rnn.pad_sequence(Y_train,
                                                             batch_first=True).float()

            sequence_lengths_train = sequence_lengths_train - horizon
            train_dataset = AutoregressiveForecastDataset(X_train_tensor,
                                                          Y_train_tensor,
                                                          sequence_lengths_train)

            X_calibration, Y_calibration, sequence_lengths_calibration = \
                generate_autoregressive_forecast_dataset(
                    n_samples=n_calibration,
                    seq_len=length,
                    horizon=horizon,
                    periodicity=periodicity,
                    amplitude=amplitude,
                    X_mean=mean,
                    X_variance=variance,
                    memory_factor=memory_factor,
                    noise_mode=noise_mode,
                    noise_profile=noise_profile,
                    dynamic_sequence_lengths=dynamic_sequence_lengths)

            # X: [n_samples, max_seq_len, n_features]
            X_calibration_tensor = torch.nn.utils.rnn.pad_sequence(X_calibration,
                                                                   batch_first=True).float()

            # Y: [n_samples, horizon, n_features]
            Y_calibration_tensor = torch.nn.utils.rnn.pad_sequence(Y_calibration,
                                                                   batch_first=True).float()

            sequence_lengths_calibration = sequence_lengths_calibration - horizon
            calibration_dataset = AutoregressiveForecastDataset(
                X_calibration_tensor,
                Y_calibration_tensor,
                sequence_lengths_calibration)

            X_test, Y_test, sequence_lengths_test = \
                generate_autoregressive_forecast_dataset(
                    n_samples=n_test,
                    seq_len=length,
                    horizon=horizon,
                    periodicity=periodicity,
                    amplitude=amplitude,
                    X_mean=mean,
                    X_variance=variance,
                    memory_factor=memory_factor,
                    noise_mode=noise_mode,
                    noise_profile=noise_profile,
                    dynamic_sequence_lengths=dynamic_sequence_lengths)

            # X: [n_samples, max_seq_len, n_features]
            X_test_tensor = torch.nn.utils.rnn.pad_sequence(X_test,
                                                            batch_first=True).float()

            # Y: [n_samples, horizon, n_features]
            Y_test_tensor = torch.nn.utils.rnn.pad_sequence(Y_test,
                                                            batch_first=True).float()

            sequence_lengths_test = sequence_lengths_train - horizon
            test_dataset = AutoregressiveForecastDataset(X_test_tensor,
                                                         Y_test_tensor,
                                                         sequence_lengths_test)

            with open('processed_data/synthetic_{}_conformal_{}.pkl'.format(
                    noise_mode, i),
                      'wb') as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset),
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            with open('processed_data/synthetic_{}_raw_{}.pkl'.format(
                    noise_mode, i),
                      'wb') as f:
                pickle.dump(((X_train, Y_train), None, (X_test, Y_test)),
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            if conformal:
                datasets.append((train_dataset, calibration_dataset, test_dataset))
            else:
                datasets.append(((X_train, Y_train), None, (X_test, Y_test)))

    return datasets
