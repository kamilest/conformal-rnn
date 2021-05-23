import pickle

import torch

from models.conformal import ConformalForecaster
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.mimic_data_processing import get_mimic_splits
from utils.performance import evaluate_performance


def run_mimic_experiments(params=None, baselines=None, retrain=False,
                          horizon=2):
    if baselines is None:
        baselines = ["CPRNN", "QRNN", "DPRNN"]
    models = {"CPRNN": ConformalForecaster, "DPRNN": DPRNN, "QRNN": QRNN}

    if params is None:
        params = {'epochs': 1000,
                  'batch_size': 150,
                  'embedding_size': 20,
                  'coverage': 0.9,
                  'lr': 0.01,
                  'n_steps': 1000,
                  'input_size': 1}

    baseline_results = dict({"CPRNN": {}, "QRNN": {}, "DPRNN": {}})

    params['max_steps'] = 49 - horizon
    params['output_size'] = horizon

    if retrain:
        for baseline in baselines:
            print('Training {}'.format(baseline))
            if baseline == 'CPRNN':
                params['epochs'] = 1000
                model = ConformalForecaster(
                    embedding_size=params['embedding_size'],
                    horizon=horizon,
                    error_rate=1 - params['coverage'])

                train_dataset, calibration_dataset, test_dataset = \
                    get_mimic_splits(conformal=True, horizon=horizon)

                model.fit(train_dataset, calibration_dataset,
                          epochs=params['epochs'], lr=params['lr'],
                          batch_size=params['batch_size'])
                coverages, intervals = model.evaluate_coverage(test_dataset)
                mean_coverage = torch.mean(coverages.float(), dim=0).item()
                interval_widths = \
                    (intervals[:, 1] - intervals[:, 0]) \
                        .squeeze().mean(dim=0).tolist()
                results = {'coverages': coverages,
                           'intervals': intervals,
                           'mean_coverage': mean_coverage,
                           'interval_widths': interval_widths}

            else:
                params['epochs'] = 10
                model = models[baseline](**params)

                train_dataset, _, test_dataset = \
                    get_mimic_splits(conformal=False, horizon=horizon)

                model.fit(train_dataset[0], train_dataset[1])
                results = evaluate_performance(model, test_dataset[0],
                                               test_dataset[1],
                                               coverage=params['coverage'],
                                               error_threshold="Auto")

            baseline_results[baseline] = results
            with open('saved_results/mimic_{}.pkl'.format(baseline), 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            for baseline in baselines:
                with open('saved_results/mimic_{}.pkl'.format(baseline),
                          'rb') as f:
                    results = pickle.load(f)
                baseline_results[baseline] = results

    return baseline_results
