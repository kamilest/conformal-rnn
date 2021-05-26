import pickle

import torch

from models.cprnn import CPRNN
from utils.data_processing_synthetic import get_synthetic_splits


def train_conformal_forecaster(noise_mode='time-dependent',
                               epochs=1000,  # LSTM parameters
                               batch_size=100,
                               embedding_size=20,
                               coverage=0.9,
                               lr=0.01,
                               retrain=False):

    if retrain:
        if noise_mode == 'periodic':
            horizon = 10
        else:
            horizon = 5

        datasets = get_synthetic_splits(noise_mode=noise_mode, conformal=True)
        results = []

        for dataset in datasets:
            train_dataset, calibration_dataset, test_dataset = dataset

            model = CPRNN(embedding_size=embedding_size, horizon=horizon,
                          error_rate=1 - coverage)
            model.fit(train_dataset, calibration_dataset, epochs=epochs, lr=lr,
                      batch_size=batch_size)

            independent_coverages, joint_coverages, intervals = \
                model.evaluate_coverage(test_dataset)
            mean_independent_coverage = torch.mean(
                independent_coverages.float(),
                dim=0)
            mean_joint_coverage = torch.mean(joint_coverages.float(),
                                             dim=0).item()
            interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()
            point_predictions, errors = \
                model.get_point_predictions_and_errors(test_dataset)

            result = {'Point predictions': point_predictions,
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
                      'Mean confidence interval widths': interval_widths.mean(
                          dim=0)}

            results.append(result)

            with open('saved_results/{}_{}.pkl'.format(noise_mode, 'CPRNN'),
                      'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('saved_results/{}_{}.pkl'.format(noise_mode, 'CPRNN'),
                  'rb') as f:
            results = pickle.load(f)

    return results
