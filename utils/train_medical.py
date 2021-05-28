import pickle

import torch

from models.cprnn import CPRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_covid import get_covid_splits
from utils.data_processing_eeg import get_eeg_splits
from utils.data_processing_mimic import get_mimic_splits
from utils.performance import evaluate_performance


def run_medical_experiments(params=None, baselines=None, retrain=False,
                            dataset='mimic', length=None, horizon=None,
                            correct_conformal=True, save_model=False,
                            save_results=True, rnn_mode='LSTM', seed=None):

    torch.manual_seed(0 if seed is None else seed)

    if seed is not None:
        retrain = True

    if baselines is None:
        baselines = ["CPRNN", "QRNN", "DPRNN"]
    models = {"CPRNN": CPRNN, "DPRNN": DPRNN, "QRNN": QRNN}

    horizons = {'mimic': 2,
                'eeg': 10,
                'covid': 50}
    if horizon is None:
        horizon = horizons[dataset]

    lengths = {'mimic': 49 - horizon,
               'eeg': 40,
               'covid': 100}
    if length is None:
        length = lengths[dataset]

    split_fns = {'mimic': get_mimic_splits,
                 'eeg': get_eeg_splits,
                 'covid': get_covid_splits}

    split_fn = split_fns[dataset]

    if params is None:
        params = {'epochs': 1000,
                  'batch_size': 150,
                  'embedding_size': 20,
                  'coverage': 0.9,
                  'lr': 0.01,
                  'n_steps': 1000,
                  'input_size': 1}

    baseline_results = dict({"CPRNN": {}, "QRNN": {}, "DPRNN": {}})

    params['max_steps'] = length
    params['output_size'] = horizon

    if retrain:
        for baseline in baselines:
            print('Training {}'.format(baseline))
            if baseline == 'CPRNN':
                params['epochs'] = 100 if dataset == 'eeg' else 1000
                model = CPRNN(
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
            if save_results:
                corr = '_uncorrected' if not correct_conformal else ''
                with open('saved_results/{}_{}{}.pkl'.format(dataset,
                                                             baseline, corr),
                          'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        for baseline in baselines:
            corr = '_uncorrected' if (baseline == 'CPRNN' and not
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
