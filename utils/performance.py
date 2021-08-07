# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors, Ahmed M. Alaa
# Licensed under the BSD 3-clause license

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------

from matplotlib import pyplot as plt

from models.bjrnn import RNN_uncertainty_wrapper
from models.dprnn import DPRNN
from models.qrnn import QRNN
from models.rnn import RNN
from utils.data_processing_synthetic import *


def evaluate_cornn_performance(model, test_dataset, correct_conformal=True,
                               normalised=True):
    independent_coverages, joint_coverages, intervals = \
        model.evaluate_coverage(
            test_dataset, corrected=correct_conformal, normalised=normalised)
    mean_independent_coverage = torch.mean(
        independent_coverages.float(),
        dim=0)
    mean_joint_coverage = torch.mean(joint_coverages.float(),
                                     dim=0).item()
    interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()
    point_predictions, errors = \
        model.get_point_predictions_and_errors(test_dataset,
                                               corrected=correct_conformal,
                                               normalised=normalised)

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


def plot_1D_uncertainty(results, Y_test, data_index):
    plt.fill_between(list(range(len(results["Lower limit"][data_index]))),
                     results["Lower limit"][data_index].reshape(-1, ),
                     results["Upper limit"][data_index].reshape(-1, ),
                     color="r", alpha=0.25)

    plt.plot(results["Lower limit"][data_index], linestyle=":", linewidth=3,
             color="r")
    plt.plot(results["Upper limit"][data_index], linestyle=":", linewidth=3,
             color="r")

    plt.plot(Y_test[data_index], linestyle="--", linewidth=2, color="black")
    plt.plot(results["Point predictions"][data_index], linewidth=3, color="r",
             Marker="o")


def get_bjrnn_coverage(intervals_, target, coverage_mode='joint'):
    lower, upper = intervals_[0], intervals_[1]

    horizon_coverages = np.logical_and(target >= lower, target <= upper)
    if coverage_mode == 'independent':
        return horizon_coverages
    else:  # joint coverage
        return np.all(horizon_coverages, axis=0)


def evaluate_bjrnn_performance(model, X_test, Y_test):
    coverages = []
    intervals = []

    for j, (x, y) in enumerate(zip(X_test, Y_test)):
        y_pred, y_l_approx, y_u_approx = model.predict(x)
        interval = np.array([y_l_approx[0], y_u_approx[0]])
        covers = get_bjrnn_coverage(interval, y.flatten().detach().numpy())
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

    return result


def evaluate_performance(model, X_test, Y_test, coverage=.9, error_threshold=1):
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
    target = np.array(Y_test)

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


def collect_synthetic_results(noise_vars, params, coverage=0.9, seq_len=5,
                              n_train_seq=1000, n_test_seq=1000):
    # noise_profs     = [noise_vars[k] * np.ones(seq_len) for k in range(len(
    # noise_vars))]
    noise_profs = noise_vars * np.ones(seq_len)

    result_dict = dict({"BJRNN": [], "QRNN": [], "DPRNN": []})

    model_type = [RNN, QRNN, DPRNN]
    model_names = ["BJRNN", "QRNN", "DPRNN"]

    for u in range(len(model_type)):

        X, Y = create_autoregressive_data(n_samples=n_train_seq,
                                          noise_profile=noise_profs,
                                          seq_len=seq_len,
                                          mode="time-dependent")

        RNN_model = model_type[u](**params)

        print("Training model " + model_names[
            u] + " with aleatoric noise variance %.4f and %d training "
                 "sequences" % (
                  noise_vars, n_train_seq))

        RNN_model.fit(X, Y)

        if type(RNN_model) is RNN:

            RNN_model_ = RNN_uncertainty_wrapper(RNN_model)

        else:

            RNN_model_ = RNN_model

        X_test, Y_test = create_autoregressive_data(n_samples=n_test_seq,
                                                    noise_profile=noise_profs,
                                                    seq_len=seq_len,
                                                    mode="time-dependent")

        result_dict[model_names[u]].append(
            evaluate_performance(RNN_model_, X_test, Y_test, coverage=coverage,
                                 error_threshold="Auto"))

    return result_dict
