# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import gc
import pickle

import torch

from models.bjrnn import RNN_uncertainty_wrapper
from models.cfrnn import CFRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from models.rnn import RNN
from utils.data_processing_synthetic import \
    EXPERIMENT_MODES, HORIZONS, generate_raw_sequences, get_synthetic_dataset, \
    MAX_SEQUENCE_LENGTHS
from utils.performance import evaluate_performance, evaluate_cfrnn_performance

CONFORMAL_FORECASTER_NAME = 'CFRNN'

DEFAULT_SYNTHETIC_PARAMS = {'input_size': 1,  # RNN parameters
                            'epochs': 10,
                            'n_steps': 500,
                            'batch_size': 100,
                            'embedding_size': 20,
                            'max_steps': 10,
                            'output_size': 5,
                            'coverage': 0.9,
                            'lr': 0.01,
                            'mode': 'LSTM'}

BASELINES = [CONFORMAL_FORECASTER_NAME, 'BJRNN', 'QRNN', 'DPRNN']

BASELINE_CLASSES = {'CFRNN': CFRNN,
                    'DPRNN': DPRNN,
                    'QRNN': QRNN}


def get_max_steps(train_dataset, test_dataset):
    return max(max(train_dataset[2]), max(test_dataset[2]))


def run_synthetic_experiments(parameters=None, baselines=None, retrain=False,
                              generate_datasets=True,
                              experiment='time-dependent',
                              correct_conformal=True, save_model=False,
                              save_results=True, rnn_mode='RNN', seed=None):
    # Models
    baselines = BASELINES if baselines is None else \
        baselines
    for baseline in baselines:
        assert baseline in BASELINES, 'Invalid baselines'

    # Datasets
    assert experiment in EXPERIMENT_MODES.keys(), 'Invalid experiment'

    # TODO move normalised to the top of file as separate baseline
    baseline_results = dict({CONFORMAL_FORECASTER_NAME: [],
                             CONFORMAL_FORECASTER_NAME + '-normalised': [],
                             'BJRNN': [],
                             'QRNN': [],
                             'DPRNN': []})

    if seed is not None:
        retrain = True
    torch.manual_seed(0 if seed is None else seed)

    if retrain:
        raw_sequence_datasets = generate_raw_sequences(experiment=experiment,
                                                       cached=(not
                                                               generate_datasets),
                                                       seed=seed)
        for baseline in baselines:
            print('Training {}'.format(baseline))

            for i, raw_sequence_dataset in enumerate(raw_sequence_datasets):
                print('Training dataset {}'.format(i))
                # Parameters
                params = DEFAULT_SYNTHETIC_PARAMS.copy() \
                    if parameters is None else parameters

                if experiment == 'long-horizon':
                    params['horizon'] = EXPERIMENT_MODES[experiment][i]
                else:
                    params['horizon'] = HORIZONS[experiment]

                params['output_size'] = params['horizon']

                if baseline == CONFORMAL_FORECASTER_NAME:
                    params['epochs'] = 1000

                    train_dataset, calibration_dataset, test_dataset = \
                        get_synthetic_dataset(raw_sequence_dataset,
                                              conformal=True, seed=seed)
                    model = CFRNN(embedding_size=params['embedding_size'],
                                  horizon=params['horizon'],
                                  error_rate=1 - params['coverage'],
                                  mode=rnn_mode)
                    model.fit(train_dataset, calibration_dataset,
                              epochs=params['epochs'], lr=params['lr'],
                              batch_size=params['batch_size'])

                    result = evaluate_cfrnn_performance(model, test_dataset,
                                                        correct_conformal)
                    result_normalised = \
                        evaluate_cfrnn_performance(model, test_dataset,
                                                   correct_conformal)
                    baseline_results[CONFORMAL_FORECASTER_NAME].append(result)
                    baseline_results[CONFORMAL_FORECASTER_NAME +
                                     '-normalised'].append(
                        result_normalised)
                else:
                    train_dataset, test_dataset = \
                        get_synthetic_dataset(raw_sequence_dataset,
                                              conformal=False, seed=seed)

                    if experiment == 'dynamic-lengths':
                        params['max_steps'] = get_max_steps(train_dataset,
                                                            test_dataset)
                    else:
                        params['max_steps'] = MAX_SEQUENCE_LENGTHS[experiment]

                    if baseline == 'BJRNN':
                        RNN_model = RNN(**params)
                        RNN_model.fit(train_dataset[0], train_dataset[1])
                        model = RNN_uncertainty_wrapper(RNN_model)
                    else:
                        model = BASELINE_CLASSES[baseline](**params)
                        model.fit(train_dataset[0], train_dataset[1])

                    result = evaluate_performance(model,
                                                  test_dataset[0],
                                                  test_dataset[1],
                                                  coverage=params['coverage'])

                    baseline_results[baseline].append(result)

                if save_model:
                    torch.save(model, 'saved_models/{}_{}_{}_{}_{}.pt'.format(
                        experiment, baseline, model.mode,
                        EXPERIMENT_MODES[experiment][i], seed))

                del model
                gc.collect()

            if save_results:
                with open('saved_results/{}_{}_{}.pkl'.format(experiment,
                                                              baseline, seed),
                          'wb') as f:
                    pickle.dump(baseline_results[baseline], f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                if baseline == CONFORMAL_FORECASTER_NAME:
                    with open('saved_results/{}_{}_{}.pkl'.format(
                            experiment, baseline + '-normalised', seed),
                            'wb') as f:
                        pickle.dump(
                            baseline_results[baseline + '-normalised'], f,
                            protocol=pickle.HIGHEST_PROTOCOL)

    else:
        for baseline in baselines:
            with open('saved_results/{}_{}_{}.pkl'.format(experiment,
                                                          baseline, seed),
                      'rb') as f:
                results = pickle.load(f)
            baseline_results[baseline] = results

    return baseline_results
