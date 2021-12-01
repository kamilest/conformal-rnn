# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.data_processing_synthetic import EXPERIMENT_MODES
from utils.train_medical import (
    CONFORMAL_BASELINES,
    get_uncorrected_medical_results,
    load_medical_results,
)
from utils.train_synthetic import load_synthetic_results


def get_joint_coverages(baseline, experiment, seeds=None):
    """ Returns joint horizon coverages for each dataset setting. """
    if seeds is None:
        seeds = list(range(5))
    coverages = []
    for seed in seeds:
        dataset_coverages = []
        results = load_synthetic_results(experiment=experiment, baseline=baseline, seed=seed)
        for result in results:
            dataset_coverages.append(result["Mean joint coverage"] * 100)
        coverages.append(dataset_coverages)
    coverages = np.array(coverages)
    return coverages.mean(axis=0), coverages.std(axis=0)


def get_joint_medical_coverages(baseline, dataset, seeds=None, correct_conformal=True):
    if seeds is None:
        seeds = list(range(5))
    coverages = []
    for seed in seeds:
        if baseline == "CFRNN" and not correct_conformal:
            result = get_uncorrected_medical_results(dataset, seed)
        else:
            result = load_medical_results(dataset=dataset, baseline=baseline, seed=seed)
        coverages.append(result["Mean joint coverage"] * 100)
    coverages = np.array(coverages)
    return coverages.mean(axis=0), coverages.std(axis=0)


def get_interval_widths(baseline, experiment, seeds=None):
    """ Returns interval widths (mean±std over seeds) across the horizon for
    every dataset setting. """

    # seeds x settings x horizon
    if seeds is None:
        seeds = list(range(5))
    widths = []
    for seed in seeds:
        results = load_synthetic_results(experiment=experiment, baseline=baseline, seed=seed)
        dataset_widths = []  # settings x horizon
        for result in results:
            # [1 x horizon] for a single dataset setting
            width = result["Mean confidence interval widths"].tolist()
            dataset_widths.append(width)
        widths.append(dataset_widths)

    widths = np.array(widths)
    # datasets (average the horizons and seeds)
    return widths.mean(axis=(0, 2)), widths.std(axis=(0, 2))


def get_medical_interval_widths(baseline, dataset, seeds=None, correct_conformal=True):
    if seeds is None:
        seeds = list(range(5))
    widths = []
    for seed in seeds:
        if baseline == "CFRNN" and not correct_conformal:
            result = get_uncorrected_medical_results(dataset, seed)
        else:
            result = load_medical_results(dataset=dataset, baseline=baseline, seed=seed)
        # [1 x horizon] for a single dataset setting
        width = result["Mean confidence interval widths"].tolist()
        widths.append(width)

    widths = np.array(widths)
    # datasets (average the horizons and seeds)
    return widths.mean(), widths.std()


def plot_timeseries(
    experiment, baseline, seed=0, index=None, forecast_only=False, figsize=(28, 4), figure_name=None, n_samples=2000
):
    assert experiment in EXPERIMENT_MODES.keys()

    plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 13})

    datasets = []
    for i in EXPERIMENT_MODES[experiment]:
        with open("processed_data/synthetic-{}-{}-{}-{}.pkl".format(experiment, i, seed, n_samples), "rb") as f:
            datasets.append(pickle.load(f))

    with open("saved_results/{}-{}-{}.pkl".format(experiment, baseline, seed), "rb") as f:
        all_results = pickle.load(f)

    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=figsize, sharey=True)

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

        ax.scatter(range(length + 1, length + horizon + 1), target, color="black")

        # Interval boundaries
        upper_limit = results["Upper limit"]
        lower_limit = results["Lower limit"]

        lower = [sequence[-1].item()] + lower_limit[i].flatten().tolist()
        upper = [sequence[-1].item()] + upper_limit[i].flatten().tolist()
        preds = [sequence[-1].item()] + results["Point predictions"][i].flatten().tolist()

        ax.fill_between(range(length, length + horizon + 1), lower, upper, color="r", alpha=0.25)
        ax.plot(range(length, length + horizon + 1), lower, linestyle="--", color="r")
        ax.plot(range(length, length + horizon + 1), upper, linestyle="--", color="r")
        ax.plot(range(length, length + horizon + 1), preds, linestyle="--", linewidth=3, color="r")

        if j == 0:
            ax.set(ylabel="Prediction")
        ax.set(xlabel="Time step")
        if experiment == "time_dependent":
            ax.set(title="$\sigma_t^2 =${:.1f}$t$".format((j + 1) * 0.1))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.4)

    if figure_name is not None:
        plt.savefig("{}.png".format(figure_name), bbox_inches="tight", dpi=600)
    plt.show()


def plot_sample_complexity(seed=0, figure_name=None):
    coverages_mean, coverages_std = {}, {}
    for baseline in ["QRNN", "DPRNN", "CFRNN"]:
        coverages_mean[baseline], coverages_std[baseline] = get_joint_coverages(
            baseline, "sample_complexity", seeds=[seed]
        )

    widths_mean = {}
    for baseline in ["QRNN", "DPRNN", "CFRNN"]:
        widths_mean[baseline] = get_interval_widths(baseline, "sample_complexity", seeds=[seed])[0].mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    figa = sns.lineplot(ax=ax1, data=coverages_mean, legend=None)
    figa.axhline(90, linestyle="--", color="black")
    figa.set(xlabel="log(Training dataset size)", ylabel="Joint coverage, %")

    figb = sns.lineplot(ax=ax2, data=widths_mean)
    figb.set(xlabel="log(Training dataset size)", ylabel="Average interval width")

    if figure_name is not None:
        plt.savefig("{}.png".format(figure_name), bbox_inches="tight", dpi=600)
    plt.show()
