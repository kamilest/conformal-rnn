# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

covid_root = "data/ltla_2021-05-24.csv"


class COVIDDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, sequence_lengths):
        super(COVIDDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]


def get_raw_covid_data(cached=True):
    if cached:
        with open("data/covid.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        df = pd.read_csv("data/ltla_2021-05-24.csv")
        for area_code in df["areaCode"].unique():
            dataset.append(
                df.loc[df["areaCode"] == area_code].sort_values("date")["newCasesByPublishDate"].to_numpy()[-250:-100]
            )
        dataset = np.array(dataset)
        with open("data/covid.pkl", "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


def get_covid_splits(
    length=100, horizon=50, conformal=True, n_train=200, n_calibration=100, n_test=80, cached=True, seed=None
):
    if seed is None:
        seed = 0
    else:
        cached = False

    if cached:
        if conformal:
            with open("processed_data/covid_conformal.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
        else:
            with open("processed_data/covid_raw.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
    else:
        raw_data = get_raw_covid_data(cached=cached)
        X = raw_data[:, :length]
        Y = raw_data[:, length : length + horizon]

        perm = np.random.RandomState(seed=seed).permutation(n_train + n_calibration + n_test)
        train_idx = perm[:n_train]
        calibration_idx = perm[n_train : n_train + n_calibration]
        train_calibration_idx = perm[: n_train + n_calibration]
        test_idx = perm[n_train + n_calibration :]

        if conformal:
            X_train = X[train_idx]
            X_calibration = X[calibration_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_calibration_scaled = scaler.transform(X_calibration)

            train_dataset = COVIDDataset(
                torch.FloatTensor(X_train_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[train_idx]).reshape(-1, horizon, 1),
                torch.ones(len(train_idx), dtype=torch.int) * length,
            )

            calibration_dataset = COVIDDataset(
                torch.FloatTensor(X_calibration_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[calibration_idx]).reshape(-1, horizon, 1),
                torch.ones(len(calibration_idx)) * length,
            )

            test_dataset = COVIDDataset(
                torch.FloatTensor(X_test_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[test_idx]).reshape(-1, horizon, 1),
                torch.ones(len(X_test_scaled), dtype=torch.int) * length,
            )

            with open("processed_data/covid_conformal.pkl", "wb") as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            X_train = X[train_calibration_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            train_dataset = X_train_scaled, Y[train_calibration_idx]
            calibration_dataset = None
            test_dataset = X_test_scaled, Y[test_idx]

            with open("processed_data/covid_raw.pkl", "wb") as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open("processed_data/covid_test_vis.pkl", "wb") as f:
            pickle.dump((X_test, Y[test_idx]), f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, calibration_dataset, test_dataset
