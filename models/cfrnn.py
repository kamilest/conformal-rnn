# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import torch


def coverage(intervals, target):
    """
    Determines whether intervals cover the target prediction.
    Depending on the coverage_mode (either 'joint' or 'independent), will return
    either a list of whether each target or all targets satisfy the coverage.

    intervals: shape [batch_size, 2, horizon, n_outputs]
    """

    lower, upper = intervals[:, 0], intervals[:, 1]
    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)
    # [batch, horizon, n_outputs], [batch, n_outputs]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


def get_critical_scores(calibration_scores, q):
    return torch.tensor([[torch.quantile(
        position_calibration_scores,
        q=q)
        for position_calibration_scores in feature_calibration_scores]
        for feature_calibration_scores in calibration_scores]).T


def get_lengths_mask(sequences, lengths, horizon):
    """Returns the lengths mask indicating the positions where every
    sequences in the batch are valid."""

    lengths_mask = torch.zeros(sequences.size(0), horizon,
                               sequences.size(2))
    for i, l in enumerate(lengths):
        lengths_mask[i, :min(l, horizon), :] = 1

    return lengths_mask


class AuxiliaryForecaster(torch.nn.Module):
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1,
                 rnn_mode='LSTM'):
        super(AuxiliaryForecaster, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.output_size = output_size

        self.rnn_mode = rnn_mode
        if self.rnn_mode == 'RNN':
            self.forecaster_rnn = torch.nn.RNN(input_size=input_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        elif self.rnn_mode == 'GRU':
            self.forecaster_rnn = torch.nn.GRU(input_size=input_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        else:  # self.mode == 'LSTM'
            self.forecaster_rnn = torch.nn.LSTM(input_size=input_size,
                                                hidden_size=embedding_size,
                                                batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size,
                                              horizon * output_size)

    def forward(self, x, state=None):
        if state is not None:
            h_0, c_0 = state
        else:
            h_0 = None

        # [batch, horizon, output_size]
        if self.rnn_mode == "LSTM":
            _, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
        else:
            _, h_n = self.forecaster_rnn(x.float(), h_0)
            c_n = None

        out = self.forecaster_out(h_n).reshape(-1, self.horizon,
                                               self.output_size)

        return out, (h_n, c_n)

    def fit(self, train_dataset, batch_size, epochs, lr):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                out, _ = self(sequences)
                valid_out = out * get_lengths_mask(sequences, lengths,
                                                   self.horizon)

                loss = criterion(valid_out.float(), targets.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 50 == 0:
                print(
                    'Epoch: {}\tTrain loss: {}'.format(epoch, mean_train_loss))

            # TODO save auxiliary forecaster to path


class CFRNN:
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1,
                 error_rate=0.05, rnn_mode='LSTM',
                 auxiliary_forecaster_path=None, **kwargs):
        super(CFRNN, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.auxiliary_forecaster_path = auxiliary_forecaster_path
        if self.auxiliary_forecaster_path:
            self.auxiliary_forecaster = torch.load(auxiliary_forecaster_path)
            for param in self.auxiliary_forecaster.parameters():
                param.requires_grad = False
        else:
            self.auxiliary_forecaster = AuxiliaryForecaster(embedding_size,
                                                            input_size,
                                                            output_size,
                                                            horizon, rnn_mode)
        self.horizon = horizon
        self.alpha = error_rate
        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None

    def nonconformity(self, output, calibration_example):
        """Measures the nonconformity between output and target time series."""
        # Average MAE loss for every step in the sequence.
        target = calibration_example[1]
        return torch.nn.functional.l1_loss(output, target, reduction='none')

    def calibrate(self, calibration_dataset):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset,
                                                         batch_size=1)
        n_calibration = len(calibration_dataset)
        calibration_scores = []

        with torch.set_grad_enabled(False):
            self.auxiliary_forecaster.eval()
            for calibration_example in calibration_loader:
                sequences, targets, lengths = calibration_example
                out, _ = self.auxiliary_forecaster(sequences)
                score = self.nonconformity(out, calibration_example)
                # n_batches: [batch_size, horizon, output_size]
                calibration_scores.append(score)

        # [output_size, horizon, n_samples]
        self.calibration_scores = torch.vstack(calibration_scores).T

        # [horizon, output_size]
        q = min((n_calibration + 1.) * (1 - self.alpha) / n_calibration, 1)
        corrected_q = min((n_calibration + 1.) * (
                1 - self.alpha / self.horizon) / n_calibration, 1)

        self.critical_calibration_scores = get_critical_scores(
            calibration_scores=self.calibration_scores,
            q=q)

        # Bonferroni corrected calibration scores.
        # [horizon, output_size]
        self.corrected_critical_calibration_scores = get_critical_scores(
            calibration_scores=self.calibration_scores,
            q=corrected_q)

    def fit(self, train_dataset, calibration_dataset, epochs, lr,
            batch_size=32, **kwargs):
        if self.auxiliary_forecaster_path is None:
            # Train the multi-horizon forecaster.
            self.auxiliary_forecaster.fit(train_dataset, batch_size, epochs, lr)

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True):
        """Forecasts the time series with conformal uncertainty intervals."""
        out, hidden = self.auxiliary_forecaster(x, state)

        if not corrected:
            # [batch_size, horizon, n_outputs]
            lower = out - self.critical_calibration_scores
            upper = out + self.critical_calibration_scores
        else:
            # [batch_size, horizon, n_outputs]
            lower = out - self.corrected_critical_calibration_scores
            upper = out + self.corrected_critical_calibration_scores

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden

    def evaluate_coverage(self, test_dataset, corrected=True):
        self.auxiliary_forecaster.eval()

        independent_coverages, joint_coverages, intervals = [], [], []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            batch_intervals, _ = self.predict(sequences, corrected=corrected)
            intervals.append(batch_intervals)
            independent_coverage, joint_coverage = coverage(batch_intervals,
                                                            targets)
            independent_coverages.append(independent_coverage)
            joint_coverages.append(joint_coverage)

        # [n_samples, (1 | horizon), n_outputs] containing booleans
        independent_coverages = torch.cat(independent_coverages)
        joint_coverages = torch.cat(joint_coverages)

        # [n_samples, 2, horizon, n_outputs] containing lower and upper bounds
        intervals = torch.cat(intervals)

        return independent_coverages, joint_coverages, intervals

    def get_point_predictions_and_errors(self, test_dataset, corrected=True):
        self.auxiliary_forecaster.eval()

        point_predictions = []
        errors = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            point_prediction, _ = self.auxiliary_forecaster(sequences)
            batch_intervals, _ = self.predict(sequences, corrected=corrected)
            point_predictions.append(point_prediction)
            errors.append(torch.nn.functional.l1_loss(point_prediction,
                                                      targets,
                                                      reduction='none').squeeze())

        point_predictions = torch.cat(point_predictions)
        errors = torch.cat(errors)

        return point_predictions, errors


class CFRNN_normalised(CFRNN, torch.nn.Module):
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1,
                 error_rate=0.05, rnn_mode='LSTM', auxiliary_forecaster_path=None, beta=1):
        super(CFRNN_normalised, self).__init__(embedding_size, input_size, output_size, horizon,
                                               error_rate, rnn_mode,
                                               auxiliary_forecaster_path)

        # Normalisation network
        self.normalising_rnn = torch.nn.RNN(input_size=self.input_size,
                                            hidden_size=self.embedding_size,
                                            batch_first=True)

        self.normalising_out = torch.nn.Linear(self.embedding_size,
                                               self.horizon * self.output_size)

        self.beta = beta

    def normaliser_forward(self, sequences):
        """Returns an estimate of normalisation target ln|y - hat{y}|."""
        _, h_n = self.normalising_rnn(sequences.float())
        out = self.normalising_out(h_n).reshape(-1, self.horizon,
                                                self.output_size)
        return out

    def normalisation_score(self, sequences, lengths):
        out = self.normaliser_forward(sequences)
        return torch.exp(out) + self.beta

    def train_normaliser(self, train_dataset, batch_size, epochs):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # TODO early stopping based on validation loss of the calibration set
        for epoch in range(epochs):
            train_loss = 0.

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                # Get the RNN multi-horizon forecast.
                forecaster_out, _ = self.auxiliary_forecaster(sequences)
                lengths_mask = get_lengths_mask(sequences, lengths,
                                                self.horizon)

                # Compute normalisation target ln|y - \hat{y}|.
                normalisation_target = \
                    torch.log(torch.abs(targets - forecaster_out)) * \
                    lengths_mask

                # Normalising network estimates the normalisation target.
                out = self.normaliser_forward(sequences)
                loss = criterion(out.float(), normalisation_target.float())
                loss.backward()

                train_loss += loss.item()
                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 100 == 0:
                print(
                    'Epoch: {}\tNormalisation loss: {}'.format(epoch,
                                                               mean_train_loss))

    def nonconformity(self, output, calibration_example):
        """Measures the nonconformity between output and target time series."""
        sequence, target, length = calibration_example
        score = torch.nn.functional.l1_loss(output, target, reduction='none')
        normalised_score = score / self.normalisation_score(sequence, length)
        return normalised_score

    def fit(self, train_dataset, calibration_dataset, epochs, lr,
            normaliser_epochs=500,
            batch_size=32):
        if self.auxiliary_forecaster_path is None:
            # Train the multi-horizon forecaster.
            self.auxiliary_forecaster.fit(train_dataset, batch_size, epochs, lr)

        # Train normalisation network.
        self.train_normaliser(train_dataset, normaliser_epochs)
        self.normalising_rnn.eval()
        self.normalising_out.eval()

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True):
        """Forecasts the time series with conformal uncertainty intervals."""
        out, hidden = self.auxiliary_forecaster(x, state)

        score = self.normalisation_score(x, len(x))
        if not corrected:
            # [batch_size, horizon, n_outputs]
            # TODO make sure len(x) is correct
            lower = out - self.critical_calibration_scores * score
            upper = out + self.critical_calibration_scores * score

        else:
            # [batch_size, horizon, n_outputs]
            lower = out - \
                    self.corrected_critical_calibration_scores * score
            upper = out + \
                    self.corrected_critical_calibration_scores * score

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden


