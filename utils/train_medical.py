import pickle

import torch

from models.cprnn import CPRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_eeg import get_eeg_splits
from utils.data_processing_mimic import get_mimic_splits
from utils.performance import evaluate_performance

torch.manual_seed(1)


def run_medical_experiments(params=None, baselines=None, retrain=False,
                            dataset='mimic', length=None, horizon=None):
    if baselines is None:
        baselines = ["CPRNN", "QRNN", "DPRNN"]
    models = {"CPRNN": CPRNN, "DPRNN": DPRNN, "QRNN": QRNN}

    if horizon is None:
        horizon = 2 if dataset == 'mimic' else 10

    split_fn = get_mimic_splits if dataset == 'mimic' else get_eeg_splits

    if params is None:
        params = {'epochs': 1000,
                  'batch_size': 150,
                  'embedding_size': 20,
                  'coverage': 0.9,
                  'lr': 0.01,
                  'n_steps': 1000,
                  'input_size': 1}

    baseline_results = dict({"CPRNN": {}, "QRNN": {}, "DPRNN": {}})

    if length is None:
        params['max_steps'] = (49 - horizon) if dataset == 'mimic' else 40
    else:
        params['max_steps'] = length

    params['output_size'] = horizon

    if retrain:
        for baseline in baselines:
            print('Training {}'.format(baseline))
            if baseline == 'CPRNN':
                params['epochs'] = 1000 if dataset == 'mimic' else 100
                model = CPRNN(
                    embedding_size=params['embedding_size'],
                    horizon=horizon,
                    error_rate=1 - params['coverage'])

                train_dataset, calibration_dataset, test_dataset = \
                    split_fn(conformal=True, horizon=horizon)

                model.fit(train_dataset, calibration_dataset,
                          epochs=params['epochs'], lr=params['lr'],
                          batch_size=params['batch_size'])
                independent_coverages, joint_coverages, intervals = \
                    model.evaluate_coverage(
                        test_dataset)
                mean_independent_coverage = torch.mean(
                    independent_coverages.float(),
                                                       dim=0)
                mean_joint_coverage = torch.mean(joint_coverages.float(),
                                                 dim=0).item()
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
                           'Mean confidence interval widths':
                               interval_widths.mean(dim=0)}

            else:
                params['epochs'] = 10
                model = models[baseline](**params)

                train_dataset, _, test_dataset = \
                    split_fn(conformal=False, horizon=horizon)

                model.fit(train_dataset[0], train_dataset[1])
                results = evaluate_performance(model, test_dataset[0],
                                               test_dataset[1],
                                               coverage=params['coverage'],
                                               error_threshold="Auto")

            baseline_results[baseline] = results
            with open('saved_results/{}_{}.pkl'.format(dataset, baseline),
                      'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        for baseline in baselines:
            with open('saved_results/{}_{}.pkl'.format(dataset, baseline),
                      'rb') as f:
                results = pickle.load(f)
            baseline_results[baseline] = results

    return baseline_results


def run_eeg_experiments(retrain=False):
    return run_medical_experiments(dataset='eeg', retrain=retrain)


def run_mimic_experiments(retrain=False):
    return run_medical_experiments(dataset='mimic', retrain=retrain)
