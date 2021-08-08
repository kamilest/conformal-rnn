# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors, Ahmed M. Alaa
# Licensed under the BSD 3-clause license

import pickle

import numpy as np
import torch

# Settings controlling the independent variables of experiments depending on
# the experiment mode:
#   periodic: Controls periodicity.
#   time-dependent: Controls increasing noise amplitude within a single time-series.
#   static: Controls noise amplitudes across the collection of time-series.
#   long-horizon: Controls the horizon length of time-series.
# See paper for details.

EXPERIMENT_MODES = {
    'periodic': [2, 10],
    'time-dependent': range(1, 6),
    'static': range(1, 6),
    'long-horizon': [5, 10, 100],
}

HORIZONS = {
    'periodic': 10,
    'time-dependent': 5,
    'static': 5,
    'long-horizon': [5, 10, 100]
}

MAX_SEQUENCE_LENGTHS = {
    'periodic': 20,
    'time-dependent': 10,
    'static': 10,
    'long-horizon': 10
}


def autoregressive(X_gen, w):
    """ Generates the autoregressive component of a single time series
    example. """
    return np.array(
        [np.sum(X_gen[0:k + 1] * np.flip(w[0:k + 1]).reshape(-1, 1)) for k in
         range(len(X_gen))])


# https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_seasonal.html
def seasonal(duration, periodicity, amplitude=1., harmonics=1,
             random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(0)

    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = amplitude * random_state.randn(harmonics)
    gamma_star_jt = amplitude * random_state.randn(harmonics)

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
                                 + amplitude * random_state.randn())
            gamma_star_jtp1[j - 1] = (- gamma_jt[j - 1] * sin_j
                                      + gamma_star_jt[j - 1] * cos_j
                                      + amplitude * random_state.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1

    return series[-duration:].reshape(-1, 1)  # Discard burn in


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
                                             horizon=10,
                                             random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(0)

    seq_len = max(seq_len, horizon)

    if noise_profile is None:
        noise_profile = [0.2, 0.4, 0.6, 0.8, 1.]

    if dynamic_sequence_lengths:
        sequence_lengths = horizon + seq_len // 2 \
                           + random_state.geometric(p=2 / seq_len,
                                                    size=n_samples)
    else:
        sequence_lengths = np.array([seq_len + horizon] * n_samples)

    # Create the input features of the generating process
    X_gen = [random_state.normal(X_mean, X_variance, (seq_len,
                                                      n_features))
             for seq_len in sequence_lengths]

    w = np.array([memory_factor ** k for k in range(np.max(sequence_lengths))])

    if noise_mode == 'static':
        noise_vars = [
            [noise_profile[(s * len(noise_profile)) // len(sequence_lengths)]] *
            sequence_lengths[s] for s in range(len(sequence_lengths))]
    elif noise_mode == 'none':
        # No additional noise beyond the variance of X_gen
        noise_vars = [[0] * sl for sl in sequence_lengths]
    else:  # noise_mode == 'time-dependent' or noise_mode == 'long-horizon'
        # Spread the noise profile across time-steps
        noise_vars = [[noise_profile[(s * len(noise_profile)) // sl]
                       for s in range(sl)] for sl in sequence_lengths]

    # X_full stores the time series values generated from features X_gen.
    ar = [autoregressive(x, w).reshape(-1, n_features) for x in X_gen]
    noise = [random_state.normal(0., nv).reshape(-1, n_features) for
             nv in noise_vars]

    if periodicity is not None:
        periodic = [seasonal(sl, periodicity, amplitude, harmonics,
                             random_state=random_state) for
                    sl in
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


def generate_raw_sequences(length=10, horizon=5,
                           n_train=2000, n_test=500,
                           cached=True,
                           mean=1,
                           variance=2,
                           memory_factor=0.9,
                           experiment='long-horizon',
                           seed=0):
    # Time series parameters
    periodicity = None
    amplitude = 1
    dynamic_sequence_lengths = False

    if cached:
        raw_sequences = []
        for i in EXPERIMENT_MODES[experiment]:
            with open('processed_data/synthetic_{}_raw_seq_{}.pkl'.format(
                    experiment, i),
                    'rb') as f:
                raw_train_sequences, raw_test_sequences = \
                    pickle.load(f)
            raw_sequences.append((raw_train_sequences, raw_test_sequences))
    else:
        raw_sequences = []
        random_state = np.random.RandomState(seed)

        for i in EXPERIMENT_MODES[experiment]:
            if experiment == 'time-dependent':
                noise_profile = [0.1 * i * k for k in range(length + horizon)]
            elif experiment == 'static':
                noise_profile = [0.1 * i for _ in range(length + horizon)]
            elif experiment == 'long-horizon':
                noise_profile = [0.1 * k for k in range(length + horizon)]
                mean = 1
                variance = 1
                horizon = 100
            else:  # noise_mode == 'periodic':
                noise_profile = [0.5 * k for k in range(length + horizon)]
                length = 20
                horizon = 10
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
                    noise_profile=noise_profile,
                    dynamic_sequence_lengths=dynamic_sequence_lengths,
                    random_state=random_state)
            sequence_lengths_train = sequence_lengths_train - horizon

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
                    noise_mode=experiment,
                    noise_profile=noise_profile,
                    dynamic_sequence_lengths=dynamic_sequence_lengths,
                    random_state=random_state)
            sequence_lengths_test = sequence_lengths_test - horizon

            with open('processed_data/synthetic_{}_raw_seq_{}.pkl'.format(
                    experiment, i),
                    'wb') as f:
                pickle.dump(((X_train, Y_train, sequence_lengths_train),
                             (X_test, Y_test, sequence_lengths_test)),
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            raw_sequences.append(((X_train, Y_train, sequence_lengths_train),
                                  (X_test, Y_test, sequence_lengths_test)))

    return raw_sequences


def get_synthetic_dataset(raw_sequences, conformal=True, n_calibration=0.5,
                          seed=0):
    (X_train, Y_train, sequence_lengths_train), \
    (X_test, Y_test, sequence_lengths_test) = raw_sequences

    if conformal:
        (X_train, Y_train, sequence_lengths_train), \
        (X_calibration, Y_calibration, sequence_lengths_calibration) = \
            split_train_dataset(X_train, Y_train, sequence_lengths_train,
                                n_calibration, seed=seed)

        # X: [n_samples, max_seq_len, n_features]
        X_train_tensor = torch.nn.utils.rnn.pad_sequence(X_train,
                                                         batch_first=True).float()

        # Y: [n_samples, horizon, n_features]
        Y_train_tensor = torch.nn.utils.rnn.pad_sequence(Y_train,
                                                         batch_first=True).float()

        train_dataset = AutoregressiveForecastDataset(X_train_tensor,
                                                      Y_train_tensor,
                                                      sequence_lengths_train)

        X_calibration_tensor = torch.nn.utils.rnn.pad_sequence(
            X_calibration,
            batch_first=True).float()

        # Y: [n_samples, horizon, n_features]
        Y_calibration_tensor = torch.nn.utils.rnn.pad_sequence(
            Y_calibration, batch_first=True).float()

        calibration_dataset = AutoregressiveForecastDataset(
            X_calibration_tensor,
            Y_calibration_tensor,
            sequence_lengths_calibration)

        # X: [n_samples, max_seq_len, n_features]
        X_test_tensor = torch.nn.utils.rnn.pad_sequence(X_test,
                                                        batch_first=True).float()

        # Y: [n_samples, horizon, n_features]
        Y_test_tensor = torch.nn.utils.rnn.pad_sequence(Y_test,
                                                        batch_first=True).float()

        test_dataset = AutoregressiveForecastDataset(X_test_tensor,
                                                     Y_test_tensor,
                                                     sequence_lengths_test)

        synthetic_dataset = (train_dataset, calibration_dataset,
                             test_dataset)
    else:
        synthetic_dataset = (X_train, Y_train), (X_test, Y_test)

    return synthetic_dataset


def split_train_dataset(X_train, Y_train, sequence_lengths_train,
                        n_calibration, seed=0):
    """ Splits the train dataset into training and calibration sets. """
    n_train = len(X_train)
    idx_perm = np.random.RandomState(seed).permutation(n_train)
    idx_calibration = idx_perm[:int(n_train * n_calibration)]
    idx_train = idx_perm[int(n_train * n_calibration):]

    X_calibration = [X_train[i] for i in idx_calibration]
    Y_calibration = [Y_train[i] for i in idx_calibration]
    sequence_lengths_calibration = [sequence_lengths_train[i] for i in
                                    idx_calibration]

    X_train = [X_train[i] for i in idx_train]
    Y_train = [Y_train[i] for i in idx_train]
    sequence_lengths_train = [sequence_lengths_train[i] for i in
                              idx_train]

    return (X_train, Y_train, sequence_lengths_train), \
           (X_calibration, Y_calibration, sequence_lengths_calibration)
