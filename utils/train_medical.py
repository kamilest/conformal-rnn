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
from utils.performance import evaluate_performance, evaluate_cornn_performance


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
    BASELINES.DPRNN: {'mimic': 10, 'eeg': 10, 'covid': 10},
    BASELINES.QRNN: {'mimic': 10, 'eeg': 10, 'covid': 10}
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

    if retrain:
        for baseline in baselines:
            print('Training {}'.format(baseline))

            conformal = baseline == BASELINES.CoRNN
            train_dataset, calibration_dataset, test_dataset = \
                split_fn(conformal=conformal, horizon=horizon, seed=seed)

            params['epochs'] = EPOCHS[baseline][dataset]

            if conformal:
                model = CoRNN(
                    embedding_size=params['embedding_size'],
                    horizon=horizon,
                    error_rate=1 - params['coverage'],
                    mode=rnn_mode)

                model.fit(train_dataset, calibration_dataset,
                          epochs=params['epochs'], lr=params['lr'],
                          batch_size=params['batch_size'])

                results = evaluate_cornn_performance(model, test_dataset,
                                                     correct_conformal)

            else:
                model = BASELINE_CLASSES[baseline](**params)

                model.fit(train_dataset[0], train_dataset[1])

                results = evaluate_performance(model, test_dataset[0],
                                               test_dataset[1],
                                               coverage=params['coverage'],
                                               error_threshold='Auto')

            baseline_results[baseline] = results

            if save_model and conformal:
                torch.save(model, 'saved_models/{}_{}_{}.pt'.format(dataset,
                                                                    baseline,
                                                                    rnn_mode))
            if save_results:
                corr = '_uncorrected' if not correct_conformal else ''
                with open('saved_results/{}_{}{}.pkl'.format(dataset,
                                                             baseline, corr),
                          'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        for baseline in baselines:
            corr = '_uncorrected' if (baseline == BASELINES.CoRNN
                                      and not correct_conformal) else ''
            with open('saved_results/{}_{}{}.pkl'.format(dataset, baseline,
                                                         corr),
                      'rb') as f:
                results = pickle.load(f)
            baseline_results[baseline] = results

    return baseline_results
