# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import pickle

import torch

from models.cfrnn import CFRNN, CFRNN_normalised
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_covid import get_covid_splits
from utils.data_processing_eeg import get_eeg_splits
from utils.data_processing_mimic import get_mimic_splits
from utils.performance import evaluate_performance, evaluate_cfrnn_performance


BASELINES = {'CFRNN': CFRNN,
             'CFRNN_normalised': CFRNN_normalised,
             'DPRNN': DPRNN,
             'QRNN': QRNN}

CONFORMAL_BASELINES = ['CFRNN', 'CFRNN_normalised']

DEFAULT_PARAMS = {'batch_size': 150,
                  'embedding_size': 20,
                  'coverage': 0.9,
                  'lr': 0.01,
                  'n_steps': 1000,
                  'input_size': 1,
                  'rnn_mode': 'LSTM'}

# Epochs are counted differently in DPRNN and QRNN compared to CoRNN but
# similar number of iterations are performed; see implementation details.
EPOCHS = {
    'CFRNN': {'mimic': 1000, 'eeg': 100, 'covid': 1000},
    'CFRNN_normalised': {'mimic': 1000, 'eeg': 100, 'covid': 1000},
    'DPRNN': {'mimic': 10, 'eeg': 10, 'covid': 10},
    'QRNN': {'mimic': 10, 'eeg': 10, 'covid': 10}
}

DATASET_SPLIT_FUNCTIONS = {'mimic': get_mimic_splits,
                           'eeg': get_eeg_splits,
                           'covid': get_covid_splits}

HORIZON_LENGTHS = {'mimic': 2,
                   'eeg': 10,
                   'covid': 50}

TIMESERIES_LENGTHS = {'mimic': 47,  # 49 - horizon
                      'eeg': 40,
                      'covid': 100}


def run_medical_experiments(dataset='mimic', baselines=None, retrain=False,
                            params=None, correct_conformal=True,
                            save_model=True, save_results=True,
                            seed=0):
    # Models
    baselines = BASELINES.keys() if baselines is None else baselines
    for baseline in baselines:
        assert baseline in BASELINES.keys(), 'Invalid baselines'

    # Datasets
    assert dataset in DATASET_SPLIT_FUNCTIONS.keys(), 'Invalid dataset'

    split_fn = DATASET_SPLIT_FUNCTIONS[dataset]
    horizon = HORIZON_LENGTHS[dataset]
    length = TIMESERIES_LENGTHS[dataset]

    # Parameters
    params = DEFAULT_PARAMS.copy() if params is None else params
    params['max_steps'] = length
    params['output_size'] = horizon

    baseline_results = {baseline: {} for baseline in BASELINES.keys()}

    torch.manual_seed(seed)

    if retrain:
        for baseline in baselines:
            print('Training {}'.format(baseline))

            conformal = baseline in CONFORMAL_BASELINES
            train_dataset, calibration_dataset, test_dataset = \
                split_fn(conformal=conformal, horizon=horizon, seed=seed)

            params['epochs'] = EPOCHS[baseline][dataset]

            if conformal:
                model = BASELINES[baseline](
                    embedding_size=params['embedding_size'],
                    horizon=horizon,
                    error_rate=1 - params['coverage'],
                    mode=params['rnn_mode'])

                results = evaluate_cfrnn_performance(model, test_dataset,
                                                     correct_conformal)

            else:
                model = BASELINES[baseline](**params)

                model.fit(train_dataset[0], train_dataset[1])

                results = evaluate_performance(model,
                                               test_dataset[0],
                                               test_dataset[1],
                                               coverage=params['coverage'])

            baseline_results[baseline] = results

            if save_model:
                torch.save(model, 'saved_models/{}-{}-{}-{}.pt'
                           .format(dataset, baseline, params['rnn_mode'], seed))
            if save_results:
                corr = '_uncorrected' if not correct_conformal else ''
                with open('saved_results/{}-{}{}-{}.pkl'.format(dataset,
                                                                baseline, corr,
                                                                seed),
                          'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        for baseline in baselines:
            corr = '_uncorrected' if (baseline in CONFORMAL_BASELINES
                                      and not correct_conformal) else ''
            with open('saved_results/{}-{}{}-{}.pkl'.format(dataset, baseline,
                                                            corr, seed),
                      'rb') as f:
                results = pickle.load(f)
            baseline_results[baseline] = results

    return baseline_results
