import torch


def nonconformity(output, target):
    """Measures the nonconformity between output and target time series."""
    # Average MAE loss for every step in the sequence.
    return torch.nn.functional.l1_loss(output, target, reduction='none')


def coverage(intervals, target, coverage_mode='joint'):
    """ Determines whether intervals coverage the target prediction.

    Depending on the coverage_mode (either 'joint' or 'independent), will return
    either a list of whether each target or all targets satisfy the coverage.

    intervals: shape [batch_size, 2, horizon, n_outputs]
    """

    lower, upper = intervals[:, 0], intervals[:, 1]

    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)

    if coverage_mode == 'independent':
        # [batch, horizon, n_outputs]
        return horizon_coverages
    else:  # joint coverage
        # [batch, n_outputs]
        return torch.all(horizon_coverages, dim=1)


class ConformalForecaster(torch.nn.Module):
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1,
                 error_rate=0.05):
        super(ConformalForecaster, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.

        # Encoder and forecaster can be the same (if embeddings are
        # trained on `horizon`-step forecasts), but different models are
        # possible.

        # TODO try separate encoder and forecaster models.
        # TODO try the RNN autoencoder trained on reconstruction error.
        self.encoder = None

        self.forecaster_rnn = torch.nn.LSTM(input_size=input_size,
                                            hidden_size=embedding_size,
                                            batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size,
                                              horizon * output_size)

        self.horizon = horizon
        self.output_size = output_size
        self.alpha = error_rate

        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None

    def forward(self, x, state=None):
        # [batch, horizon, output_size]
        _, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
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

                loss = criterion(valid_out, targets)
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

    def fit(self, train_dataset, calibration_dataset, epochs, lr,
            batch_size=32):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # Train the multi-horizon forecaster.
        self.train_forecaster(train_loader, epochs, lr)
        # Collect calibration scores
        self.calibrate(calibration_dataset, n_train=len(train_dataset))

    def predict(self, x, state=None, coverage_mode='joint'):
        """Forecasts the time series with conformal uncertainty intervals."""
        # TODO +/- nonconformity will not return *adaptive* interval widths.
        out, hidden = self(x, state)

        if coverage_mode == 'independent':
            # [batch_size, horizon, n_outputs]
            lower = out - self.critical_calibration_scores
            upper = out + self.critical_calibration_scores
        else:
            # [batch_size, horizon, n_outputs]
            lower = out - self.corrected_critical_calibration_scores
            upper = out + self.corrected_critical_calibration_scores

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden

    def evaluate_coverage(self, test_dataset, coverage_mode='joint'):
        self.eval()

        coverages, intervals = [], []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            batch_intervals, _ = self.predict(sequences)
            intervals.append(batch_intervals)
            coverages.append(coverage(batch_intervals, targets,
                                      coverage_mode=coverage_mode))

        # [n_samples, (1 | horizon), n_outputs] containing booleans
        coverages = torch.cat(coverages)

        # [n_samples, 2, horizon, n_outputs] containing lower and upper bounds
        intervals = torch.cat(intervals)

        return coverages, intervals
