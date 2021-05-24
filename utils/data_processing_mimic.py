# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class MIMICDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, sequence_lengths):
        super(MIMICDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]


def process_mimic_data(horizon=2, feature='wbchigh'):
    feature_names = ['temphigh', 'heartratehigh', 'sysbplow', 'diasbplow',
                     'meanbplow', 'spo2high',
                     'fio2high', 'respratelow', 'glucoselow', 'bicarbonatehigh',
                     'bicarbonatelow', 'creatininehigh', 'creatininelow',
                     'hematocrithigh',
                     'hematocritlow', 'hemoglobinhigh', 'hemoglobinlow',
                     'platelethigh',
                     'plateletlow', 'potassiumlow', 'potassiumhigh', 'bunhigh',
                     'bunlow',
                     'wbchigh', 'wbclow', 'antibiotics', 'norepinephrine',
                     'mechanical_ventilator'
                     'age', 'weight']

    idx = feature_names.index(feature)

    with open('data/mimic.p', 'rb') as f:
        MIMIC_data = pickle.load(f)

    Y = MIMIC_data["longitudinal"][:, :, idx]  # 'wbchigh'
    L = MIMIC_data['trajectory_lengths']

    X_ = []
    Y_ = []
    L_ = []
    for k in np.where(L > 4)[0]:
        X_.append(Y[k, :L[k] - horizon])
        Y_.append(Y[k, L[k] - horizon:L[k]])
        L_.append(L[k])

    return X_, Y_, L_


def get_mimic_dataset(X, Y, L, idx):
    X_ = torch.nn.utils.rnn.pad_sequence([torch.FloatTensor(X[i]).reshape(-1, 1)
                                          for i in idx],
                                         batch_first=True)
    Y_ = torch.nn.utils.rnn.pad_sequence([torch.FloatTensor(Y[i]).reshape(-1, 1)
                                          for i in idx],
                                         batch_first=True).float()

    return MIMICDataset(X_, Y_, [L[i] for i in idx])


def get_mimic_splits(n_train=2000, n_calibration=1823, n_test=500,
                     conformal=True, feature='wbchigh', horizon=2):
    perm = np.random.RandomState(seed=0).permutation(n_train + n_calibration +
                                                     n_test)
    train_idx = perm[:n_train]
    calibration_idx = perm[n_train:n_train + n_calibration]
    train_calibration_idx = perm[:n_train + n_calibration]
    test_idx = perm[n_train + n_calibration:]

    X_, Y_, L = process_mimic_data(feature=feature, horizon=horizon)
    assert n_train + n_calibration + n_test == len(X_)

    # TODO scaling

    if conformal:
        train_dataset = get_mimic_dataset(X_, Y_, L, train_idx)
        calibration_dataset = get_mimic_dataset(X_, Y_, L, calibration_idx)
        test_dataset = get_mimic_dataset(X_, Y_, L, test_idx)

    else:
        train_dataset = [X_[i] for i in train_calibration_idx], \
                        [Y_[i] for i in train_calibration_idx],
        calibration_dataset = None
        test_dataset = [X_[i] for i in test_idx], [Y_[i] for i in test_idx]

    # TODO save test dataset for visualisation
    # with open('processed_data/mimic_test.pkl', 'wb') as f:
    #     pickle.dump((X_test, Y_test), f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, calibration_dataset, test_dataset
