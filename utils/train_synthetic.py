import torch

from models.cprnn import CPRNN
from utils.make_data import generate_autoregressive_forecast_dataset


def train_conformal_forecaster(n_train_samples=1000,
                               n_calibration_samples=1000,
                               n_test_samples=500,

                               # Time series parameters
                               mean=1,
                               variance=2,
                               memory_factor=0.9,
                               noise_mode='time-dependent',
                               seq_len=10,
                               horizon=5,
                               noise_profile=None,
                               periodicity=None,
                               amplitude=1,
                               dynamic_sequence_lengths=False,

                               # LSTM parameters
                               epochs=1000,
                               batch_size=100,
                               embedding_size=20,
                               coverage=0.9,
                               lr=0.01):
    if noise_profile is None:
        noise_profile = [0.1 * k for k in range(seq_len + horizon)]

    train_dataset = generate_autoregressive_forecast_dataset(
        n_samples=n_train_samples,
        seq_len=seq_len,
        horizon=horizon,
        periodicity=periodicity,
        amplitude=amplitude,
        X_mean=mean,
        X_variance=variance,
        memory_factor=memory_factor,
        noise_mode=noise_mode,
        noise_profile=noise_profile,
        dynamic_sequence_lengths=dynamic_sequence_lengths)

    calibration_dataset = generate_autoregressive_forecast_dataset(
        n_samples=n_calibration_samples,
        seq_len=seq_len,
        horizon=horizon,
        periodicity=periodicity,
        amplitude=amplitude,
        X_mean=mean,
        X_variance=variance,
        memory_factor=memory_factor,
        noise_mode=noise_mode,
        noise_profile=noise_profile,
        dynamic_sequence_lengths=dynamic_sequence_lengths)

    test_dataset = generate_autoregressive_forecast_dataset(
        n_samples=n_test_samples,
        seq_len=seq_len,
        horizon=horizon,
        periodicity=periodicity,
        amplitude=amplitude,
        X_mean=mean,
        X_variance=variance,
        memory_factor=memory_factor,
        noise_mode=noise_mode,
        noise_profile=noise_profile,
        dynamic_sequence_lengths=dynamic_sequence_lengths)

    model = CPRNN(embedding_size=embedding_size, horizon=horizon,
                  error_rate=1 - coverage)
    model.fit(train_dataset, calibration_dataset, epochs=epochs, lr=lr,
              batch_size=batch_size)

    independent_coverages, joint_coverages, intervals = \
        model.evaluate_coverage(test_dataset)
    mean_independent_coverage = torch.mean(independent_coverages.float(), dim=0)
    mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()
    interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()
    point_predictions, errors = \
        model.get_point_predictions_and_errors(test_dataset)
    results = {'Point predictions': point_predictions,
               'Errors': errors,
               'Independent coverage indicators':
                   independent_coverages.squeeze(),
               'Joint coverage indicators':
                   joint_coverages.squeeze(),
               'Upper limit': intervals[:, 1],
               'Lower limit': intervals[:, 0],
               'Mean independent coverage':
                   mean_independent_coverage.squeeze(),
               'Mean joint coverage': mean_joint_coverage,
               'Confidence interval widths': interval_widths,
               'Mean confidence interval widths': interval_widths.mean(dim=0)}

    return results
