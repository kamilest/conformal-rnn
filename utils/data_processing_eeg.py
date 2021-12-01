# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import gzip
import os
import pathlib
import pickle

import numpy as np
import torch
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

eeg_root_train = "data/eeg/SMNI_CMI_TRAIN"
eeg_root_test = "data/eeg/SMNI_CMI_TEST"


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, sequence_lengths):
        super(EEGDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]


def parse_eeg_file(filename):
    with gzip.open(filename, "rb") as f:
        chans = {}
        for line in f:
            tokens = line.decode("ascii").split()
            if tokens[0] != "#":
                if tokens[1] not in chans.keys():
                    chans[tokens[1]] = []
                chans[tokens[1]].append(float(tokens[3]))
        chan_arrays = []
        for chan in chans.values():
            chan_arrays.append(chan)
    return chan_arrays


def get_raw_eeg_data(split="train", include_alcoholic_class=False, cached=True):
    if split == "train":
        root = eeg_root_train
    else:
        root = eeg_root_test

    if cached:
        with open("data/eeg_{}.pkl".format(split), "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        for folder in os.listdir(root):
            if folder != "README" and (include_alcoholic_class or folder[3] == "c"):
                subfolder = os.path.join(root, folder)
                for filename in os.listdir(subfolder):
                    f = os.path.join(subfolder, filename)
                    if ".gz" in pathlib.Path(f).suffixes:
                        chan_arrays = parse_eeg_file(f)
                        dataset.extend(chan_arrays)
        with open("data/eeg_{}.pkl".format(split), "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


def downsample_sequences(raw_data, length, horizon):
    raw_data_resampled = resample(raw_data, length + horizon, axis=1)

    X = raw_data_resampled[:, :-horizon]
    Y = raw_data_resampled[:, -horizon:]
    return X, Y


def get_eeg_splits(length=40, horizon=10, calibrate=0.2, conformal=True, cached=True, seed=None):
    if seed is None:
        seed = 0
    else:
        cached = False

    if cached:
        if conformal:
            with open("processed_data/eeg_conformal.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
        else:
            with open("processed_data/eeg_raw.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
    else:
        X_train, Y_train = downsample_sequences(get_raw_eeg_data("train"), length, horizon)
        X_test, Y_test = downsample_sequences(get_raw_eeg_data("test"), length, horizon)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if conformal:
            calibration_idx = np.random.RandomState(seed=seed).choice(
                len(X_train), replace=False, size=int(calibrate * len(X_train))
            )
            train_idx = np.setdiff1d(range(len(X_train)), calibration_idx)

            train_dataset = EEGDataset(
                torch.FloatTensor(X_train_scaled[train_idx]).reshape(-1, length, 1),
                torch.FloatTensor(Y_train[train_idx]).reshape(-1, horizon, 1),
                torch.ones(len(train_idx), dtype=torch.int) * length,
            )

            calibration_dataset = EEGDataset(
                torch.FloatTensor(X_train_scaled[calibration_idx]).reshape(-1, length, 1),
                torch.FloatTensor(Y_train[calibration_idx]).reshape(-1, horizon, 1),
                torch.ones(len(calibration_idx)) * length,
            )

            test_dataset = EEGDataset(
                torch.FloatTensor(X_test_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y_test).reshape(-1, horizon, 1),
                torch.ones(len(X_test_scaled), dtype=torch.int) * length,
            )

            with open("processed_data/eeg_conformal.pkl", "wb") as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            train_dataset = X_train_scaled, Y_train
            calibration_dataset = None
            test_dataset = X_test_scaled, Y_test

            with open("processed_data/eeg_raw.pkl", "wb") as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open("processed_data/eeg_test_vis.pkl", "wb") as f:
            pickle.dump((X_test, Y_test), f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, calibration_dataset, test_dataset
