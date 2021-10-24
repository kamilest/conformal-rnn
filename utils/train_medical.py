# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

import pickle

import torch

from models.cfrnn import CFRNN, AdaptiveCFRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_covid import get_covid_splits
from utils.data_processing_eeg import get_eeg_splits
from utils.data_processing_mimic import get_mimic_splits
from utils.performance import evaluate_performance, evaluate_cfrnn_performance

BASELINES = {'CFRNN': CFRNN,
             'AdaptiveCFRNN': AdaptiveCFRNN,
             'DPRNN': DPRNN,
             'QRNN': QRNN}

CONFORMAL_BASELINES = ['CFRNN', 'AdaptiveCFRNN']

DEFAULT_MEDICAL_PARAMETERS = {'batch_size': 150,
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
    'AdaptiveCFRNN': {'mimic': 1000, 'eeg': 100, 'covid': 1000},
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


def run_medical_experiments(dataset, baseline,
                            params=None,
                            save_model=False, save_results=True,
                            seed=0):
    assert baseline in BASELINES.keys(), 'Invalid baselines'
    assert dataset in DATASET_SPLIT_FUNCTIONS.keys(), 'Invalid dataset'

    split_fn = DATASET_SPLIT_FUNCTIONS[dataset]
    horizon = HORIZON_LENGTHS[dataset]
    length = TIMESERIES_LENGTHS[dataset]

    # Parameters
    params = DEFAULT_MEDICAL_PARAMETERS.copy() if params is None else params
    params['max_steps'] = length
    params['output_size'] = horizon
    params['epochs'] = EPOCHS[baseline][dataset]

    torch.manual_seed(seed)

    print('Training {}'.format(baseline))

    if baseline in CONFORMAL_BASELINES:
        train_dataset, calibration_dataset, test_dataset = \
            split_fn(conformal=True, horizon=horizon, seed=seed)

        model = BASELINES[baseline](
            embedding_size=params['embedding_size'],
            horizon=horizon,
            error_rate=1 - params['coverage'],
            mode=params['rnn_mode'])

        model.fit(train_dataset, calibration_dataset,
                  epochs=params['epochs'], lr=params['lr'],
                  batch_size=params['batch_size'],)

        results = evaluate_cfrnn_performance(model, test_dataset)
    else:
        train_dataset, _, test_dataset = \
            split_fn(conformal=False, horizon=horizon, seed=seed)

        model = BASELINES[baseline](**params)
        model.fit(train_dataset[0], train_dataset[1])
        results = evaluate_performance(model,
                                       test_dataset[0],
                                       test_dataset[1],
                                       coverage=params['coverage'])

    if save_model:
        torch.save(model, 'saved_models/{}-{}-{}.pt'
                   .format(dataset, baseline, seed))
    if save_results:
        with open('saved_results/{}-{}-{}.pkl'.format(dataset,
                                                        baseline,
                                                        seed),
                  'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results


def get_uncorrected_medical_results(dataset, seed):
    with open('saved_models/{}-{}-{}.pt'.format(dataset, 'CFRNN', seed),
              'rb') as f:
        model = torch.load(f)

    split_fn = DATASET_SPLIT_FUNCTIONS[dataset]
    horizon = HORIZON_LENGTHS[dataset]
    _, _, test_dataset = \
        split_fn(conformal=True, horizon=horizon, seed=seed)

    results = evaluate_cfrnn_performance(model, test_dataset,
                                         correct_conformal=False)
    return results


def load_medical_results(dataset, baseline, seed):
    with open('saved_results/{}-{}-{}.pkl'.format(dataset, baseline, seed),
              'rb') as f:
        results = pickle.load(f)
    return results
