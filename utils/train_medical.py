# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import pickle
from enum import Enum

import torch

from models.cornn import CoRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_covid import get_covid_splits
from utils.data_processing_eeg import get_eeg_splits
from utils.data_processing_mimic import get_mimic_splits
from utils.performance import evaluate_performance


class BASELINES(Enum):
    CoRNN = 'CPRNN'
    DPRNN = 'DPRNN'
    QRNN = 'QRNN'


BASELINE_CLASSES = {BASELINES.CoRNN: CoRNN,
                    BASELINES.DPRNN: DPRNN,
                    BASELINES.QRNN: QRNN}

DEFAULT_PARAMS = {'batch_size': 150,
                  'embedding_size': 20,
                  'coverage': 0.9,
                  'lr': 0.01,
                  'n_steps': 1000,
                  'input_size': 1}

# Epochs are counted differently in DPRNN and QRNN compared to CoRNN but
# similar number of iterations are performed; see implementation details.
EPOCHS = {
    BASELINES.CoRNN: {'mimic': 1000, 'eeg': 100, 'covid': 1000},
    BASELINES.DPRNN: {'mimic': 100, 'eeg': 10, 'covid': 10},
    BASELINES.QRNN: {'mimic': 1000, 'eeg': 10, 'covid': 10}
}


DATASET_SPLIT_FNS = {'mimic': get_mimic_splits,
                     'eeg': get_eeg_splits,
                     'covid': get_covid_splits}

HORIZON_LENGTHS = {'mimic': 2,
                   'eeg': 10,
                   'covid': 50}

TS_LENGTHS = {'mimic': 47,  # 49 - horizon
              'eeg': 40,
              'covid': 100}


def run_medical_experiments(params=None, baselines=None, retrain=False,
                            dataset='mimic', length=None, horizon=None,
                            correct_conformal=True, save_model=False,
                            save_results=True, rnn_mode='LSTM', seed=None):
    # Models
    baselines = BASELINE_CLASSES.keys() if baselines is None else baselines
    for baseline in baselines:
        assert baseline in BASELINE_CLASSES.keys(), 'Invalid baselines'

    # Datasets
    assert dataset in DATASET_SPLIT_FNS.keys(), 'Invalid dataset'
    split_fn = DATASET_SPLIT_FNS[dataset]
    horizon = HORIZON_LENGTHS[dataset] if horizon is None else horizon
    length = TS_LENGTHS[dataset] if length is None else length

    # Parameters
    params = DEFAULT_PARAMS if params is None else params
    params['max_steps'] = length
    params['output_size'] = horizon

    baseline_results = dict({BASELINES.CoRNN: {}, BASELINES.QRNN: {},
                             BASELINES.DPRNN: {}})

    torch.manual_seed(0 if seed is None else seed)
    if seed is not None:
        retrain = True

    def train_cornn():
        model = CoRNN(
            embedding_size=params['embedding_size'],
            horizon=horizon,
            error_rate=1 - params['coverage'],
            mode=rnn_mode)

        train_dataset, calibration_dataset, test_dataset = \
            split_fn(conformal=True, horizon=horizon, seed=seed)

        model.fit(train_dataset, calibration_dataset,
                  epochs=params['epochs'], lr=params['lr'],
                  batch_size=params['batch_size'])
        if save_model:
            torch.save(model, 'saved_models/{}_{}_{}.pt'.format(dataset,
                                                                baseline,
                                                                model.mode))
        independent_coverages, joint_coverages, intervals = \
            model.evaluate_coverage(
                test_dataset, corrected=correct_conformal)
        mean_independent_coverage = torch.mean(
            independent_coverages.float(),
            dim=0)
        mean_joint_coverage = torch.mean(joint_coverages.float(),
                                         dim=0).item()
        interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()
        point_predictions, errors = \
            model.get_point_predictions_and_errors(test_dataset,
                                                   corrected=correct_conformal)

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
                   'Mean confidence interval widths':
                       interval_widths.mean(dim=0)}

        return results

    if retrain:
        for baseline in baselines:
            print('Training {}'.format(baseline))
            params['epochs'] = EPOCHS[baseline][dataset]

            if baseline == BASELINES.CoRNN:
                results = train_cornn()

            else:
                model = BASELINE_CLASSES[baseline](**params)

                train_dataset, _, test_dataset = \
                    split_fn(conformal=False, horizon=horizon)

                model.fit(train_dataset[0], train_dataset[1])
                results = evaluate_performance(model, test_dataset[0],
                                               test_dataset[1],
                                               coverage=params['coverage'],
                                               error_threshold='Auto')

            baseline_results[baseline] = results
            if save_results:
                corr = '_uncorrected' if not correct_conformal else ''
                with open('saved_results/{}_{}{}.pkl'.format(dataset,
                                                             baseline, corr),
                          'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        for baseline in baselines:
            corr = '_uncorrected' if (baseline == BASELINES.CoRNN and not
            correct_conformal) else ''
            with open('saved_results/{}_{}{}.pkl'.format(dataset, baseline,
                                                         corr),
                      'rb') as f:
                results = pickle.load(f)
            baseline_results[baseline] = results

    return baseline_results


def run_eeg_experiments(retrain=False):
    return run_medical_experiments(dataset='eeg', retrain=retrain)


def run_mimic_experiments(retrain=False):
    return run_medical_experiments(dataset='mimic', retrain=retrain)
