import pickle

import numpy as np
import pandas as pd
import torch


def get_windows(X, length, stride, horizon, return_tensors=False):
    n_seq = (len(X) - length - horizon) // stride
    X_ = [X[i * stride:i * stride + length] for i in range(n_seq)]
    Y_ = [X[i * stride + length:i * stride + length + horizon] for i in
          range(n_seq)]

    if return_tensors:
        return torch.tensor(X_), torch.tensor(Y_)
    else:
        return X_, Y_


def get_multi_feature_windows(df_X, length, stride, horizon,
                              return_tensors=True):
    XX = []
    YY = []
    for feature in df_X.columns:
        X_, Y_ = get_windows(df_X[feature].values, length, stride, horizon,
                             return_tensors=False)
        XX.extend(X_)
        YY.extend(Y_)

    if return_tensors:
        return torch.tensor(XX), torch.tensor(YY)
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
        return self.X[start_idx:end_idx], self.X[end_idx:end_idx + self.horizon]


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
            train_dataset = torch.utils.data.TensorDataset(*train_windows)
            cal_dataset = torch.utils.data.TensorDataset(*cal_windows)
            test_dataset = torch.utils.data.TensorDataset(*test_windows)

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

# def run_uci_experiments(retrain=False):
#     baselines = ["CoRNN", "QRNN", "DPRNN"]
#     exp_res_dict = dict({"CoRNN": [], "QRNN": [], "DPRNN": []})
#
#     baseline_results = dict({"CoRNN": dict({"CI_length": None,
#                                             "Errors": None,
#                                             "Coverages": None,
#                                             "Intervals": None}),
#                              "QRNN": dict({"CI_length": None,
#                                            "Errors": None,
#                                            "Coverages": None,
#                                            "Intervals": None}),
#                              "DPRNN": dict({"CI_length": None,
#                                             "Errors": None,
#                                             "Coverages": None,
#                                             "Intervals": None})})
#
#     if retrain:
#
#         for n_sample_ in n_samples_:
#             exp_results = collect_synthetic_results(noise_var_, params,
#                                                     coverage=coverage,
#                                                     seq_len=seq_len,
#                                                     n_train_seq=n_train_seq,
#                                                     n_test_seq=n_test_seq)
#
#             exp_res_dict["BJRNN"].append(exp_results["BJRNN"])
#             exp_res_dict["QRNN"].append(exp_results["QRNN"])
#             exp_res_dict["DPRNN"].append(exp_results["DPRNN"])
#     else:
#
#         infile = open('saved_models/Experiment_2_result', 'rb')
#         exp_res_dict = pickle.load(infile)
#
#     elif exp_mode == "1":
#
#     noise_vars = [0.1 * k for k in range(9)]
#
#     if retrain:
#
#         for noise_var in noise_vars:
#             exp_results = collect_synthetic_results(noise_var, params,
#                                                     coverage=coverage,
#                                                     seq_len=seq_len,
#                                                     n_train_seq=n_samples,
#                                                     n_test_seq=n_test)
#
#             exp_res_dict["BJRNN"].append(exp_results["BJRNN"])
#             exp_res_dict["QRNN"].append(exp_results["QRNN"])
#             exp_res_dict["DPRNN"].append(exp_results["DPRNN"])
#
#     else:
#
#         infile = open('saved_models/Experiment_1_result', 'rb')
#         exp_res_dict = pickle.load(infile)
#
#
# for baseline in baselines:
#     baseline_results[baseline]["CI_length"] = [
#         exp_res_dict[baseline][k][0]["CI length"] for k in
#         range(len(exp_res_dict[baseline]))]
#     baseline_results[baseline]["Errors"] = [
#         np.mean(np.concatenate(exp_res_dict[baseline][k][0]["Errors"])) for
#         k in range(len(exp_res_dict[baseline]))]
#     baseline_results[baseline]["Coverages"] = BJRNN_coverages = [
#         exp_res_dict[baseline][k][0]["Coverage"] for k in
#         range(len(exp_res_dict[baseline]))]
#
# return baseline_results
