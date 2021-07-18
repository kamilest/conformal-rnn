import torch


def nonconformity(output, target):
    """Measures the nonconformity between output and target time series."""
    # Average MAE loss for every step in the sequence.
    return torch.nn.functional.l1_loss(output, target, reduction='none')


def coverage(intervals, target):
    """ Determines whether intervals coverage the target prediction.

    Depending on the coverage_mode (either 'joint' or 'independent), will return
    either a list of whether each target or all targets satisfy the coverage.

    intervals: shape [batch_size, 2, horizon, n_outputs]
    """

    lower, upper = intervals[:, 0], intervals[:, 1]

    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)

    # [batch, horizon, n_outputs], [batch, n_outputs]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


class CoRNN(torch.nn.Module):
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1,
                 error_rate=0.05, mode='LSTM', **kwargs):
        super(CoRNN, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.

        # Encoder and forecaster can be the same (if embeddings are
        # trained on `horizon`-step forecasts), but different models are
        # possible.

        # TODO try separate encoder and forecaster models.
        # TODO try the RNN autoencoder trained on reconstruction error.
        self.encoder = None

        self.mode = mode
        if self.mode == 'RNN':
            self.forecaster_rnn = torch.nn.RNN(input_size=input_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        elif self.mode == 'GRU':
            self.forecaster_rnn = torch.nn.GRU(input_size=input_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        else:  # self.mode == 'LSTM'
            self.forecaster_rnn = torch.nn.LSTM(input_size=input_size,
                                                hidden_size=embedding_size,
                                                batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size,
                                              horizon * output_size)

        self.horizon = horizon
        self.output_size = output_size
        self.alpha = error_rate

        self.n_train = None
        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None

    def forward(self, x, state=None):
        if state is not None:
            h_0, c_0 = state
        else:
            h_0 = None

        # [batch, horizon, output_size]
        if self.mode == "LSTM":
            _, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
        else:
            _, h_n = self.forecaster_rnn(x.float(), h_0)
            c_n = None

        out = self.forecaster_out(h_n).reshape(-1, self.horizon,
                                               self.output_size)

        return out, (h_n, c_n)

    def train_forecaster(self, train_loader, epochs, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                out, _ = self(sequences)

                lengths_mask = torch.zeros(sequences.size(0), self.horizon,
                                           sequences.size(2))
                for i, l in enumerate(lengths):
                    lengths_mask[i, :min(l, self.horizon), :] = 1
                valid_out = lengths_mask * out

                loss = criterion(valid_out.float(), targets.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 50 == 0:
                print(
                    'Epoch: {}\tTrain loss: {}'.format(epoch, mean_train_loss))

    def calibrate(self, calibration_dataset, n_train):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset,
                                                         batch_size=1)
        calibration_scores = []

        with torch.set_grad_enabled(False):
            self.eval()
            for sequences, targets, lengths in calibration_loader:
                out, _ = self(sequences)
                # n_batches: [batch_size, horizon, output_size]
                calibration_scores.append(nonconformity(out, targets))

        # [output_size, horizon, n_samples]
        self.calibration_scores = torch.vstack(calibration_scores).T

        # [horizon, output_size]
        self.critical_calibration_scores = torch.tensor([[torch.quantile(
            position_calibration_scores,
            q=1 - self.alpha * n_train / (n_train + 1))
            for position_calibration_scores in feature_calibration_scores]
            for feature_calibration_scores in self.calibration_scores]).T

        # Bonferroni corrected calibration scores.
        # [horizon, output_size]
        corrected_alpha = self.alpha / self.horizon
        self.corrected_critical_calibration_scores = torch.tensor([[
            torch.quantile(
                position_calibration_scores,
                q=1 - corrected_alpha * n_train / (n_train + 1))
            for position_calibration_scores in feature_calibration_scores]
            for feature_calibration_scores in self.calibration_scores]).T

        self.n_train = n_train

    def fit(self, train_dataset, calibration_dataset, epochs, lr,
            batch_size=32):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # Train the multi-horizon forecaster.
        self.train_forecaster(train_loader, epochs, lr)
        # Collect calibration scores
        self.calibrate(calibration_dataset, n_train=len(train_dataset))

    def predict(self, x, state=None, corrected=True):
        """Forecasts the time series with conformal uncertainty intervals."""
        # TODO +/- nonconformity will not return *adaptive* interval widths.
        out, hidden = self(x, state)

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
        self.eval()

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
        self.eval()

        point_predictions = []
        errors = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            point_prediction, _ = self(sequences)
            batch_intervals, _ = self.predict(sequences, corrected=corrected)
            point_predictions.append(point_prediction)
            errors.append(torch.nn.functional.l1_loss(point_prediction,
                                                      targets,
                                                      reduction='none').squeeze())

        point_predictions = torch.cat(point_predictions)
        errors = torch.cat(errors)

        return point_predictions, errors
