# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------

import matplotlib.pyplot as plt

from models.bjrnn import RNN_uncertainty_wrapper
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.data_processing_synthetic import *


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
    target = np.array([t for t in Y_test]).squeeze()

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
