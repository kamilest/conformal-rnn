# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import gc
import pickle

import torch

from models.bjrnn import RNN_uncertainty_wrapper
from models.cfrnn import CFRNN, AdaptiveCFRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from models.rnn import RNN
from utils.data_processing_synthetic import \
    EXPERIMENT_MODES, get_raw_sequences, get_synthetic_dataset, \
    DEFAULT_PARAMETERS
from utils.performance import evaluate_performance, evaluate_cfrnn_performance

BASELINES = {'CFRNN': CFRNN,
             'AdaptiveCFRNN': AdaptiveCFRNN,
             'BJRNN': None,
             'DPRNN': DPRNN,
             'QRNN': QRNN}

CONFORMAL_BASELINES = ['CFRNN', 'AdaptiveCFRNN']

DEFAULT_SYNTHETIC_TRAINING_PARAMETERS = {'input_size': 1,  # RNN parameters
                                         'epochs': 10,
                                         'normaliser_epochs': 1000,
                                         'n_steps': 500,
                                         'batch_size': 100,
                                         'embedding_size': 20,
                                         'max_steps': 10,
                                         'horizon': 5,
                                         'coverage': 0.9,
                                         'lr': 0.01,
                                         'rnn_mode': 'LSTM',
                                         'beta': 1}


def get_max_steps(train_dataset, test_dataset):
    return max(max(train_dataset[2]), max(test_dataset[2]))


def run_synthetic_experiments(experiment, baseline,
                              retrain=False, params=None,
                              dynamic_sequence_lengths=False,
                              horizon=None, beta=None,
                              cached_datasets=True, correct_conformal=True,
                              save_model=False, save_results=True,
                              rnn_mode=None, seed=0):
    assert baseline in BASELINES.keys(), 'Invalid baseline'
    assert experiment in EXPERIMENT_MODES.keys(), 'Invalid experiment'

    baseline_results = []

    torch.manual_seed(seed)

    if retrain:
        raw_sequence_datasets = \
            get_raw_sequences(experiment=experiment,
                              cached=cached_datasets,
                              dynamic_sequence_lengths=dynamic_sequence_lengths,
                              horizon=horizon,
                              seed=seed)
        print('Training {}'.format(baseline))

        for i, raw_sequence_dataset in enumerate(raw_sequence_datasets):
            print('Training dataset {}'.format(i))

            if params is None:
                params = DEFAULT_SYNTHETIC_TRAINING_PARAMETERS.copy()

            if rnn_mode is not None:
                params['rnn_mode'] = rnn_mode

            if beta is not None:
                params['beta'] = beta

            params['output_size'] = \
                horizon if horizon else DEFAULT_PARAMETERS['horizon']

            if baseline in CONFORMAL_BASELINES:
                params['epochs'] = 1000

                train_dataset, calibration_dataset, test_dataset = \
                    get_synthetic_dataset(raw_sequence_dataset,
                                          conformal=True, seed=seed)
                model = BASELINES[baseline](
                    embedding_size=params['embedding_size'],
                    horizon=params['horizon'],
                    error_rate=1 - params['coverage'],
                    rnn_mode=params['rnn_mode'],
                    auxiliary_forecaster_path= \
                        'saved_models/{}-aux-{}-{}-{}.pt'.format(
                            experiment, params['rnn_mode'],
                            EXPERIMENT_MODES[experiment][i], seed),
                    beta=params['beta'])
                model.fit(train_dataset, calibration_dataset,
                          epochs=params['epochs'], lr=params['lr'],
                          batch_size=params['batch_size'],
                          normaliser_epochs=params['normaliser_epochs'])

                result = evaluate_cfrnn_performance(model, test_dataset,
                                                    correct_conformal)

            else:
                train_dataset, test_dataset = \
                    get_synthetic_dataset(raw_sequence_dataset,
                                          conformal=False, seed=seed)

                if dynamic_sequence_lengths or horizon is None:
                    params['max_steps'] = get_max_steps(train_dataset,
                                                        test_dataset)

                if baseline == 'BJRNN':
                    RNN_model = RNN(**params)
                    RNN_model.fit(train_dataset[0], train_dataset[1])
                    model = RNN_uncertainty_wrapper(RNN_model)
                else:
                    model = BASELINES[baseline](**params)
                    model.fit(train_dataset[0], train_dataset[1])

                result = evaluate_performance(model,
                                              test_dataset[0],
                                              test_dataset[1],
                                              coverage=params['coverage'])

            baseline_results.append(result)

            if save_model:
                torch.save(model,
                           'saved_models/{}-{}-{}-{}-{}.pt'.format(
                               experiment, baseline, model.rnn_mode,
                               EXPERIMENT_MODES[experiment][i], seed))

            del model
            gc.collect()

        if save_results:
            with open('saved_results/{}-{}-{}.pkl'.format(experiment,
                                                          baseline, seed),
                      'wb') as f:
                pickle.dump(baseline_results, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('saved_results/{}-{}-{}.pkl'.format(experiment,
                                                      baseline, seed),
                  'rb') as f:
            baseline_results = pickle.load(f)

    return baseline_results


def load_synthetic_results(experiment, baseline, seed=0):
    with open('saved_results/{}-{}-{}.pkl'.format(experiment,
                                                  baseline, seed),
              'rb') as f:
        baseline_results = pickle.load(f)

    return baseline_results
