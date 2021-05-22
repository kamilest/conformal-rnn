import pickle

import numpy as np
import pandas as pd
import torch

from models.conformal import ConformalForecaster
from models.dprnn import DPRNN
from models.qrnn import QRNN
from utils.performance import evaluate_performance


def get_windows(X, length, stride, horizon, return_tensors=False):
    n_seq = (len(X) - length - horizon) // stride
    X_ = [X[i * stride:i * stride + length] for i in range(n_seq)]
    Y_ = [X[i * stride + length:i * stride + length + horizon] for i in
          range(n_seq)]

    if return_tensors:
        return torch.FloatTensor(X_), torch.FloatTensor(Y_)
    else:
        return X_, Y_


def get_multi_feature_windows(df_X, length, stride, horizon,
                              return_tensors=True):
    XX = []
    YY = []
    for feature in df_X.columns:
        X_, Y_ = get_windows(df_X[feature].values.reshape(-1, 1), length,
                             stride,
                             horizon,
                             return_tensors=False)
        XX.extend(X_)
        YY.extend(Y_)

    if return_tensors:
        return torch.FloatTensor(XX), torch.FloatTensor(YY)
    else:
        return XX, YY


class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, X, length, stride, horizon):
        super(WindowedDataset, self).__init__()
        self.X = X
        self.length = length
        self.stride = stride
        self.horizon = horizon

    def __len__(self):
        return (len(self.X) - self.length - self.horizon) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.length
        return self.X[start_idx:end_idx], \
               self.X[end_idx:end_idx + self.horizon], \
               self.length


class HungarianChickenpoxDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, length):
        super(HungarianChickenpoxDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.length = length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.length


def get_dataset(dataset, length=None, stride=None, horizon=None,
                return_raw=False, calibrate=True):
    if dataset == 'energy' or dataset == 'stock':
        if dataset == 'energy':
            df = pd.read_csv('data/energy_data.csv')
            train_features = ['Appliances']
            length = 600 if length is None else length
            stride = 5 if stride is None else stride
            horizon = 120 if horizon is None else horizon

        else:
            df = pd.read_csv('data/stock_data.csv')
            train_features = ['Close']
            length = 30 if length is None else length
            stride = 1 if stride is None else stride
            horizon = 10 if horizon is None else horizon

        # 70%, 20%, 10% split
        n = len(df)
        if calibrate:
            train_df = df[0:int(n * 0.7)]
        else:
            train_df = df[0:int(n * 0.9)]

        test_df = df[int(n * 0.9):]

        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        X_train = train_df[train_features].values
        X_test = test_df[train_features].values

        X_cal = np.array([])
        if calibrate:
            cal_df = df[int(n * 0.7):int(n * 0.9)]
            cal_df = (cal_df - train_mean) / train_std
            X_cal = cal_df[train_features].values

        if not return_raw:
            train_dataset = WindowedDataset(X_train, length, stride, horizon)
            cal_dataset = WindowedDataset(X_cal, length, stride, horizon)
            test_dataset = WindowedDataset(X_test, length, stride, horizon)
        else:
            train_dataset = get_windows(X_train, length, stride, horizon)
            cal_dataset = get_windows(X_cal, length, stride, horizon)
            test_dataset = get_windows(X_test, length, stride, horizon)

    else:  # Hungarian chickenpox
        df = pd.read_csv('data/hungary_chickenpox/hungary_chickenpox.csv')
        length = 8 if length is None else length
        stride = 1 if stride is None else stride
        horizon = 4 if horizon is None else horizon

        if calibrate:
            train_features = ['BUDAPEST', 'BACS', 'BEKES', 'BORSOD', 'CSONGRAD',
                              'HAJDU', 'HEVES', 'JASZ',
                              'NOGRAD', 'PEST', 'SZABOLCS']
            cal_features = ['KOMAROM', 'GYOR', 'VESZPREM', 'TOLNA', 'FEJER']
        else:
            train_features = ['BUDAPEST', 'BACS', 'BEKES', 'BORSOD', 'CSONGRAD',
                              'HAJDU', 'HEVES', 'JASZ',
                              'NOGRAD', 'PEST', 'SZABOLCS', 'KOMAROM', 'GYOR',
                              'VESZPREM', 'TOLNA', 'FEJER']
            cal_features = []
        test_features = ['BARANYA', 'SOMOGY', 'ZALA', 'VAS']

        train_df = df[train_features]
        cal_df = df[cal_features]
        test_df = df[test_features]

        train_mean = train_df.mean().mean()
        train_std = train_df.std().mean()

        train_df = (train_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        cal_df = (cal_df - train_mean) / train_std

        return_tensors = not return_raw
        train_windows = get_multi_feature_windows(train_df, length,
                                                  stride, horizon,
                                                  return_tensors=return_tensors)
        cal_windows = get_multi_feature_windows(cal_df, length,
                                                stride, horizon,
                                                return_tensors=return_tensors)
        test_windows = get_multi_feature_windows(test_df, length,
                                                 stride, horizon,
                                                 return_tensors=return_tensors)
        if return_raw:
            train_dataset = train_windows
            cal_dataset = cal_windows
            test_dataset = test_windows
        else:
            train_dataset = HungarianChickenpoxDataset(*train_windows, length)
            cal_dataset = HungarianChickenpoxDataset(*cal_windows, length)
            test_dataset = HungarianChickenpoxDataset(*test_windows, length)

    return train_dataset, cal_dataset, test_dataset


def prepare_uci_datasets():
    for dataset in ['energy', 'stock', 'hungary']:
        for calibrate in [True, False]:
            print('Dataset: {}, calibrated: {}'.format(dataset, calibrate))
            return_raw = not calibrate
            calibrated = 'calibrated' if calibrate else 'raw'
            name = '{}_{}_default'.format(dataset, calibrated)
            train, cal, test = get_dataset(dataset, calibrate=calibrate,
                                           return_raw=return_raw)

            with open('data/{}_train.pkl'.format(name), 'wb') as f:
                pickle.dump(train, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            if calibrate:
                with open('data/{}_calibrate.pkl'.format(name), 'wb') as f:
                    pickle.dump(cal, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open('data/{}_test.pkl'.format(name), 'wb') as f:
                pickle.dump(test, f,
                            protocol=pickle.HIGHEST_PROTOCOL)


def run_uci_experiments(params=None, baselines=None, datasets=None,
                        retrain=False):
    if datasets is None:
        datasets = ['energy', 'stock', 'hungary']

    if baselines is None:
        baselines = ["CoRNN", "QRNN", "DPRNN"]
    models = {"CoRNN": ConformalForecaster, "DPRNN": DPRNN, "QRNN": QRNN}

    if params is None:
        params = {'epochs': 1000,
                  'batch_size': 100,
                  'embedding_size': 20,
                  'coverage': 0.9,
                  'lr': 0.01,
                  'n_steps': 500,
                  'input_size': 1}

    baseline_results = dict({"CoRNN": dict({'energy': None, 'stock': None,
                                            'hungary': None}),
                             "QRNN": dict({'energy': None, 'stock': None,
                                           'hungary': None}),
                             "DPRNN": dict({'energy': None, 'stock': None,
                                            'hungary': None})})

    for dataset in datasets:
        if dataset == 'energy':
            max_steps = 600
            output_size = horizon = 120
        elif dataset == 'stock':
            max_steps = 30
            output_size = horizon = 10
        else:
            max_steps = 8
            output_size = horizon = 4

        params['max_steps'] = max_steps
        params['output_size'] = output_size

        for baseline in baselines:
            print('{}, {}'.format(baseline, dataset))

            calibrated = 'calibrated' if baseline == 'CoRNN' else 'raw'
            name = '{}_{}_default'.format(dataset, calibrated)

            with open('data/{}_train.pkl'.format(name), 'rb') as f:
                train_dataset = pickle.load(f)
            if baseline == 'CoRNN':
                with open('data/{}_calibrate.pkl'.format(name), 'rb') as f:
                    calibration_dataset = pickle.load(f)
            else:
                calibration_dataset = None
            with open('data/{}_test.pkl'.format(name), 'rb') as f:
                test_dataset = pickle.load(f)

            if retrain:
                if baseline == 'CoRNN':
                    params['epochs'] = 1000
                    if dataset == 'energy':
                        params['epochs'] = 100
                    model = ConformalForecaster(
                        embedding_size=params['embedding_size'],
                        horizon=horizon,
                        error_rate=1 - params['coverage'])
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
                    model.fit(train_dataset[0], train_dataset[1])
                    results = evaluate_performance(model, test_dataset[0],
                                                   test_dataset[1],
                                                   coverage=params['coverage'],
                                                   error_threshold="Auto")

                with open('saved_results/{}_{}.pkl'.format(baseline, dataset),
                          'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                with open('saved_results/{}_{}.pkl'.format(baseline, dataset),
                          'rb') as f:
                    results = pickle.load(f)

            baseline_results[baseline][dataset] = results

    return baseline_results
