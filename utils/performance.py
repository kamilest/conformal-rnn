# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors, Ahmed M. Alaa
# Licensed under the BSD 3-clause license

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------

import matplotlib.pyplot as plt

from models.bjrnn import RNN_uncertainty_wrapper
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_synthetic import *
from utils.train_synthetic import load_synthetic_results


def evaluate_cfrnn_performance(model, test_dataset, correct_conformal=True):
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


def evaluate_performance(model, X_test, Y_test, coverage=.9):
    if type(model) is RNN_uncertainty_wrapper:
        y_pred, y_l_approx, y_u_approx = model.predict(X_test,
                                                       coverage=coverage)

    elif type(model) is QRNN:
        y_u_approx, y_l_approx = model.predict(X_test)
        y_pred = [(y_l_approx[k] + y_u_approx[k]) / 2 for k in
                  range(len(y_u_approx))]

        y_pred = [x.reshape(-1, 1) for x in y_pred]
        y_u_approx = [x.reshape(-1, 1) for x in y_u_approx]
        y_l_approx = [x.reshape(-1, 1) for x in y_l_approx]

    elif type(model) is DPRNN:
        y_pred, y_std = model.predict(X_test, alpha=1 - coverage)
        y_u_approx = [y_pred[k] + y_std[k] for k in range(len(y_pred))]
        y_l_approx = [y_pred[k] - y_std[k] for k in range(len(y_pred))]

        y_pred = [x.reshape(-1, 1) for x in y_pred]
        y_u_approx = [x.reshape(-1, 1) for x in y_u_approx]
        y_l_approx = [x.reshape(-1, 1) for x in y_l_approx]

    results = {}

    upper = np.array(y_u_approx).squeeze()
    lower = np.array(y_l_approx).squeeze()
    pred = np.array(y_pred).squeeze()
    target = np.array([t.numpy() for t in Y_test]).squeeze()

    results["Point predictions"] = np.array(y_pred)
    results["Upper limit"] = np.array(y_u_approx)
    results["Lower limit"] = np.array(y_l_approx)
    results["Confidence interval widths"] = upper - lower
    results["Mean confidence interval widths"] = np.mean(upper - lower, axis=0)
    results["Errors"] = np.abs(target - pred)

    independent_coverage = np.logical_and(upper >= target, lower <= target)

    results["Independent coverage indicators"] = independent_coverage
    results["Joint coverage indicators"] = np.all(independent_coverage, axis=1)
    results["Mean independent coverage"] = np.mean(independent_coverage,
                                                   axis=0)
    results["Mean joint coverage"] = np.mean(
        results["Joint coverage indicators"])

    return results


def get_joint_coverage(baseline, experiment):
    # TODO get the dataframe of coverage for seed (row) and horizon (col)
    # print(baseline)
    coverages = []
    for seed in range(5):
        dataset_setting_coverages = []
        results = load_synthetic_results(experiment=experiment,
                                         baseline=baseline, seed=seed)
        for result in results:  # for each setting
            dataset_setting_coverages.append(result['Mean joint coverage'] * 100)
        coverages.append(dataset_setting_coverages)
    coverages = np.array(coverages)
    for m, s in zip(coverages.mean(axis=0), coverages.std(axis=0)):
        print('{:.1f} \\(\\pm\\) {:.1f}\\%'.format(m, s))
    print()


def get_interval_widths(baseline, experiment):
    # TODO get interval width for and horizon (col) for an
    #  experiment accounting for the dataset settings (row) ± over seeds and
    #  return as dataframe
    pass

    print(baseline)
    wws = []  # array containing interval widths for horizon=5 for each seed
    for seed in range(5):
        ws = []
        results = load_synthetic_results(experiment=experiment,
                                         baseline=baseline, seed=seed)
        for result in results:  # for each setting
            # for the data setttings (increasing time-dependent noise)
            widths = result['Mean confidence interval widths']  # [1xhorizon]
            # averages across the horizon, elements of ws represent different experiment mode
            ws.append(
                '{:.2f} \\(\\pm\\) {:.2f}'.format(widths.mean(), widths.std()))
        wws.append(ws)

    # rows denote increasing time-dependent noise configuration, columns denote seeds
    for i in range(1):
        #         print('{} & {} & {} & {} & {}'.format(wws[0][i], wws[1][i], wws[2][i], wws[3][i], wws[4][i]))
        for j in range(5):
            print(wws[j][i])
    print()


def plot_timeseries(experiment, baseline, seed=0, index=None,
                    forecast_only=False, figsize=(28, 4), figure_name=None):
    # TODO cleanup
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({'axes.titlesize': 16,
                         'axes.labelsize': 13})

    datasets = []
    for i in ([2, 10] if experiment == 'periodic' else range(1, 6)):
        with open('processed_data/synthetic-{}-{}-{}.pkl'.format(experiment, i,
                                                                 seed),
                  'rb') as f:
            datasets.append(pickle.load(f))

    with open('saved_results/{}-{}-{}.pkl'.format(experiment, baseline, seed),
              'rb') as f:
        all_results = pickle.load(f)

    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=figsize,
                            sharey=True)

    for j, ax in enumerate(axs.flat):
        _, dataset = datasets[j]
        results = all_results[j]

        X, Y, _ = dataset

        if index is None:
            index = range(len(X))
        i = np.random.choice(index)

        # Derive unpadded sequences and targets
        sequence, target = X[i], Y[i]

        horizon = len(target)
        length = len(sequence)

        if not forecast_only:
            # (Scatter)plot of the time series
            ax.plot(range(1, length + 1), sequence, color="black")

            # Prediction start vertical
            ax.axvline(length, linestyle="--", color="black")

        ax.scatter(range(length + 1, length + horizon + 1), target,
                   color="black")

        # Interval boundaries
        upper_limit = results['Upper limit']
        lower_limit = results['Lower limit']

        lower = [sequence[-1].item()] + lower_limit[i].flatten().tolist()
        upper = [sequence[-1].item()] + upper_limit[i].flatten().tolist()
        preds = [sequence[-1].item()] + results['Point predictions'][
            i].flatten().tolist()

        ax.fill_between(range(length, length + horizon + 1), lower, upper,
                        color="r", alpha=0.25)
        ax.plot(range(length, length + horizon + 1), lower, linestyle="--",
                color="r")
        ax.plot(range(length, length + horizon + 1), upper, linestyle="--",
                color="r")
        ax.plot(range(length, length + horizon + 1), preds, linestyle="--",
                linewidth=3, color="r")

        if j == 0:
            ax.set(ylabel='Prediction')
        ax.set(xlabel='Time step')
        if experiment == 'time_dependent':
            ax.set(title='$\sigma_t^2 =${:.1f}$t$'.format((j + 1) * 0.1))

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.4)

    if figure_name is not None:
        plt.savefig('{}.png'.format(figure_name), bbox_inches='tight')
    plt.show()

# TODO compute valid horizons

# TODO compute interval widths vs coverage

# TODO demonstrate sample complexity
