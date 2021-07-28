# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import gc
import pickle

import numpy as np
import torch

from models.bjrnn import RNN_uncertainty_wrapper
from models.cornn import CoRNN
from models.rnn import RNN
from utils.data_processing_synthetic import get_synthetic_splits, \
    EXPERIMENT_MODES, HORIZONS
from utils.performance import evaluate_performance, evaluate_cornn_performance

CONFORMAL_FORECASTER_NAME = 'CPRNN'


def train_conformal_forecaster(experiment_mode='time-dependent',
                               epochs=1000,  # LSTM parameters
                               batch_size=100,
                               embedding_size=20,
                               coverage=0.9,
                               lr=0.01,
                               retrain=False,
                               save_model=False,
                               save_results=True,
                               rnn_mode='LSTM'):
    if retrain:
        datasets = get_synthetic_splits(noise_mode=experiment_mode,
                                        conformal=True)
        results = []

        for i, dataset in enumerate(datasets):
            if experiment_mode == 'long-horizon':
                horizon = EXPERIMENT_MODES[experiment_mode][i]
            else:
                horizon = HORIZONS[experiment_mode]

            train_dataset, calibration_dataset, test_dataset = dataset

            model = CoRNN(embedding_size=embedding_size, horizon=horizon,
                          error_rate=1 - coverage, mode=rnn_mode)
            model.fit(train_dataset, calibration_dataset, epochs=epochs, lr=lr,
                      batch_size=batch_size)
            if save_model:
                torch.save(model, 'saved_models/{}_{}_{}_{}.pt'.format(
                    experiment_mode, CONFORMAL_FORECASTER_NAME, model.mode,
                    EXPERIMENT_MODES[experiment_mode][i]))

            result = evaluate_cornn_performance(model, test_dataset)
            results.append(result)

        if save_results:
            with open('saved_results/{}_{}.pkl'.format(experiment_mode,
                                                       CONFORMAL_FORECASTER_NAME),
                      'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('saved_results/{}_{}.pkl'.format(experiment_mode,
                                                   CONFORMAL_FORECASTER_NAME),
                  'rb') as f:
            results = pickle.load(f)

    return results


def train_blockwise_forecaster(noise_mode='time-dependent',
                               epochs=10,  # LSTM parameters
                               batch_size=100,
                               embedding_size=20,
                               coverage=0.9,
                               lr=0.01,
                               retrain=False):
    if retrain:
        params = {'epochs': epochs,
                  'batch_size': batch_size,
                  'embedding_size': embedding_size,
                  'coverage': coverage,
                  'lr': lr,
                  'n_steps': 10,
                  'input_size': 1,
                  'mode': 'LSTM'}

        # TODO separate parameters / clear documentation
        if noise_mode == 'periodic':
            length = 20
            horizon = 10
        else:
            length = 10
            horizon = 5

        params['max_steps'] = length
        params['output_size'] = horizon

        datasets = get_synthetic_splits(conformal=False, horizon=horizon)
        results = []
        for dataset in datasets:
            train_dataset, _, test_dataset = dataset

            model = RNN(**params)
            model.fit(train_dataset[0], train_dataset[1])
            model_ = RNN_uncertainty_wrapper(model)
            result = evaluate_performance(model_, test_dataset[0],
                                          test_dataset[1],
                                          coverage=params['coverage'],
                                          error_threshold="Auto")

            results.append(result)
            del model_

            with open('saved_results/{}_{}.pkl'.format(noise_mode, 'BJRNN'),
                      'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('saved_results/{}_{}.pkl'.format(noise_mode, 'BJRNN'),
                  'rb') as f:
            results = pickle.load(f)

    return results


def train_bjrnn(noise_mode='time-dependent'):
    def get_coverage(intervals_, target, coverage_mode='joint'):
        lower, upper = intervals_[0], intervals_[1]

        horizon_coverages = np.logical_and(target >= lower, target <= upper)
        if coverage_mode == 'independent':
            return horizon_coverages
        else:  # joint coverage
            return np.all(horizon_coverages, axis=0)

    params = dict({"input_size": 1,  # RNN parameters
                   "epochs": 1000,
                   "n_steps": 5,
                   "batch_size": 100,
                   "embedding_size": 20,
                   "max_steps": 10,
                   "output_size": 5,
                   "coverage": 0.9,
                   "lr": 0.01,
                   "mode": "RNN"})

    results = []
    for i in range(1, 6):
        with open('processed_data/synthetic_{}_raw_{}.pkl'.format(
                noise_mode, i),
                'rb') as f:
            train_dataset, calibration_dataset, test_dataset = \
                pickle.load(f)
        X_train, Y_train = train_dataset
        X_test, Y_test = test_dataset

        RNN_model = RNN(**params)
        RNN_model.fit(X_train, Y_train)

        RNN_model_ = RNN_uncertainty_wrapper(RNN_model)

        coverages = []
        intervals = []

        for j, (x, y) in enumerate(zip(X_test, Y_test)):
            y_pred, y_l_approx, y_u_approx = RNN_model_.predict(x)
            interval = np.array([y_l_approx[0], y_u_approx[0]])
            covers = get_coverage(interval, y.flatten().detach().numpy())
            coverages.append(covers)
            intervals.append(interval)
            if j % 50 == 0:
                print('Example {}'.format(j))

        mean_coverage = np.mean(coverages)
        np_intervals = np.array(intervals)
        interval_widths = (np_intervals[:, 1] - np_intervals[:, 0]).mean(axis=0)

        result = {'coverages': coverages,
                  'intervals': intervals,
                  'mean_coverage': mean_coverage,
                  'interval_widths': interval_widths}

        print('Model {}:\tcoverage: {}\twidths: {}'.format(i, result[
            'mean_coverage'], result['interval_widths']))
        results.append(result)
        del RNN_model
        del RNN_model_
        gc.collect()

    with open('saved_results/{}_BJRNN.pkl'.format(noise_mode), 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

    return results
