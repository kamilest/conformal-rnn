# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

""" CFRNN model. """

import os.path

import torch


def coverage(intervals, target):
    """
    Determines whether intervals cover the target prediction
    considering each target horizon either separately or jointly.

    Args:
        intervals: shape [batch_size, 2, horizon, n_outputs]
        target: ground truth forecast values

    Returns:
        individual and joint coverage rates
    """

    lower, upper = intervals[:, 0], intervals[:, 1]
    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)
    # [batch, horizon, n_outputs], [batch, n_outputs]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


def get_critical_scores(calibration_scores, q):
    """
    Computes critical calibration scores from scores in the calibration set.

    Args:
        calibration_scores: calibration scores for each example in the
            calibration set.
        q: target quantile for which to return the calibration score

    Returns:
        critical calibration scores for each target horizon
    """

    return torch.tensor(
        [
            [
                torch.quantile(position_calibration_scores, q=q)
                for position_calibration_scores in feature_calibration_scores
            ]
            for feature_calibration_scores in calibration_scores
        ]
    ).T


def get_lengths_mask(sequences, lengths, horizon):
    """
    Returns the mask indicating which positions in the sequence are valid.

    Args:
        sequences: (batch of) input sequences
        lengths: the lengths of every sequence in the batch
        horizon: the forecasting horizon
    """

    lengths_mask = torch.zeros(sequences.size(0), horizon, sequences.size(2))
    for i, l in enumerate(lengths):
        lengths_mask[i, : min(l, horizon), :] = 1

    return lengths_mask


class AuxiliaryForecaster(torch.nn.Module):
    """
    The auxiliary RNN issuing point predictions.

    Point predictions are used as baseline to which the (normalised)
    uncertainty intervals are added in the main CFRNN network.
    """

    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1, rnn_mode="LSTM", path=None):
        """
        Initialises the auxiliary forecaster.

        Args:
            embedding_size: hyperparameter indicating the size of the latent
                RNN embeddings.
            input_size: dimensionality of the input time-series
            output_size: dimensionality of the forecast
            horizon: forecasting horizon
            rnn_mode: type of the underlying RNN network
            path: optional path where to save the auxiliary model to be used
                in the main CFRNN network
        """
        super(AuxiliaryForecaster, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.output_size = output_size
        self.path = path

        self.rnn_mode = rnn_mode
        if self.rnn_mode == "RNN":
            self.forecaster_rnn = torch.nn.RNN(input_size=input_size, hidden_size=embedding_size, batch_first=True)
        elif self.rnn_mode == "GRU":
            self.forecaster_rnn = torch.nn.GRU(input_size=input_size, hidden_size=embedding_size, batch_first=True)
        else:  # self.mode == 'LSTM'
            self.forecaster_rnn = torch.nn.LSTM(input_size=input_size, hidden_size=embedding_size, batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size, horizon * output_size)

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

        out = self.forecaster_out(h_n).reshape(-1, self.horizon, self.output_size)

        return out, (h_n, c_n)

    def fit(self, train_dataset, batch_size, epochs, lr):
        """
        Trains the auxiliary forecaster to the training dataset.

        Args:
            train_dataset: a dataset of type `torch.utils.data.Dataset`
            batch_size: batch size
            epochs: number of training epochs
            lr: learning rate
        """
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                out, _ = self(sequences)
                valid_out = out * get_lengths_mask(sequences, lengths, self.horizon)

                loss = criterion(valid_out.float(), targets.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 50 == 0:
                print("Epoch: {}\tTrain loss: {}".format(epoch, mean_train_loss))

        if self.path is not None:
            torch.save(self, self.path)


class CFRNN:
    """
    The basic CFRNN model as presented in Algorithm 1 of the accompanying paper.

    CFRNN training procedure entails training the underlying (auxiliary) model
    on the training dataset (which is implemented as part of
    `AuxiliaryForecaster`), and calibrating the predictions of the auxiliary
    forecaster against the calibration dataset.

    The calibration is done by computing the empirical distribution of
    nonconformity scores (implemented via the `nonconformity`
    function), via the `calibrate` method.

    The AuxiliaryForecaster can be fit to the dataset from scratch, or,
    if the model path is provided, the model is loaded directly, its training is
    skipped and only the calibration procedure is carried out.

    Additional methods are provided for returning predictions: on the test
    example, the point prediction is done by the underlying
    `AuxiliaryForecaster`, and the horizon-specific critical calibration scores
    (obtained from the calibration procedure) are added to the point forecast to
    obtain the resulting interval. The coverage can be further evaluated by
    comparing the uncertainty intervals to the ground truth forecasts,
    returning joint and independent coverages and getting the errors from the
    point prediction.
    """

    def __init__(
        self,
        embedding_size,
        input_size=1,
        output_size=1,
        horizon=1,
        error_rate=0.05,
        rnn_mode="LSTM",
        auxiliary_forecaster_path=None,
        **kwargs
    ):
        """
        Args:
            embedding_size: size of the embedding of the underlying point
                forecaster
            input_size: dimensionality of observed time-series
            output_size: dimensionality of a forecast step
            horizon: forecasting horizon (number of steps into the future)
            error_rate: controls the error rate for the joint coverage in the
                estimated uncertainty intervals
            rnn_mode: type of the underlying AuxiliaryForecaster model
            auxiliary_forecaster_path: training of the underlying
                `AuxiliaryForecaster` can be skipped if the path for the
                already trained `AuxiliaryForecaster` is provided.
        """
        super(CFRNN, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.rnn_mode = rnn_mode
        self.requires_auxiliary_fit = True

        self.auxiliary_forecaster_path = auxiliary_forecaster_path
        if self.auxiliary_forecaster_path and os.path.isfile(self.auxiliary_forecaster_path):
            self.auxiliary_forecaster = torch.load(auxiliary_forecaster_path)
            for param in self.auxiliary_forecaster.parameters():
                param.requires_grad = False
            self.requires_auxiliary_fit = False
        else:
            self.auxiliary_forecaster = AuxiliaryForecaster(
                embedding_size, input_size, output_size, horizon, rnn_mode, auxiliary_forecaster_path
            )
        self.horizon = horizon
        self.alpha = error_rate
        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None

    def nonconformity(self, output, calibration_example):
        """
        Measures the nonconformity between output and target time series.

        Args:
            output: the point prediction given by the auxiliary forecasting
                model
            calibration_example: the tuple consisting of calibration
                example's input sequence, ground truth forecast, and sequence
                length

        Returns:
            Average MAE loss for every step in the sequence.
        """
        # Average MAE loss for every step in the sequence.
        target = calibration_example[1]
        return torch.nn.functional.l1_loss(output, target, reduction="none")

    def calibrate(self, calibration_dataset: torch.utils.data.Dataset):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=1)
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
        q = min((n_calibration + 1.0) * (1 - self.alpha) / n_calibration, 1)
        corrected_q = min((n_calibration + 1.0) * (1 - self.alpha / self.horizon) / n_calibration, 1)

        self.critical_calibration_scores = get_critical_scores(calibration_scores=self.calibration_scores, q=q)

        # Bonferroni corrected calibration scores.
        # [horizon, output_size]
        self.corrected_critical_calibration_scores = get_critical_scores(
            calibration_scores=self.calibration_scores, q=corrected_q
        )

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        calibration_dataset: torch.utils.data.Dataset,
        epochs,
        lr,
        batch_size=32,
        **kwargs
    ):
        """
        Fits the CFRNN model.

        If the auxiliary forecaster is not trained, fits the underlying
        `AuxiliaryForecaster` on the training dataset using the batch size,
        learning rate and number of epochs provided. Otherwise, the auxiliary
        forecaster that has been loaded on initialisation is used. On fitting
        the underlying model, computes calibration scores for the calibration
        dataset.

        Args:
            train_dataset: training dataset on which the underlying
            forecasting model is trained
            calibration_dataset: calibration dataset used to compute the
            empirical nonconformity score distribution
            epochs: number of epochs for training the underlying forecaster
            lr: learning rate for training the underlying forecaster
            batch_size: batch size for training the underlying forecaster
        """

        if self.requires_auxiliary_fit:
            # Train the multi-horizon forecaster.
            self.auxiliary_forecaster.fit(train_dataset, batch_size, epochs, lr)

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True):
        """
        Forecasts the time series with conformal uncertainty intervals.

        Args:
            x: time-series to be forecasted
            state: initial state for the underlying auxiliary forecaster RNN
            corrected: whether to use Bonferroni-corrected calibration scores

        Returns:
            tensor with lower and upper forecast bounds; hidden RNN state
        """

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

    def evaluate_coverage(self, test_dataset: torch.utils.data.Dataset, corrected=True):
        """
        Evaluates coverage of the examples in the test dataset.

        Args:
            test_dataset: test dataset
            corrected: whether to use the Bonferroni-corrected critical
            calibration scores
        Returns:
            independent and joint coverages, forecast uncertainty intervals
        """
        self.auxiliary_forecaster.eval()

        independent_coverages, joint_coverages, intervals = [], [], []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            batch_intervals, _ = self.predict(sequences, corrected=corrected)
            intervals.append(batch_intervals)
            independent_coverage, joint_coverage = coverage(batch_intervals, targets)
            independent_coverages.append(independent_coverage)
            joint_coverages.append(joint_coverage)

        # [n_samples, (1 | horizon), n_outputs] containing booleans
        independent_coverages = torch.cat(independent_coverages)
        joint_coverages = torch.cat(joint_coverages)

        # [n_samples, 2, horizon, n_outputs] containing lower and upper bounds
        intervals = torch.cat(intervals)

        return independent_coverages, joint_coverages, intervals

    def get_point_predictions_and_errors(self, test_dataset: torch.utils.data.Dataset, corrected=True):
        """
        Obtains point predictions of the examples in the test dataset.

        Obtained by running the Auxiliary forecaster and adding the
        calibrated uncertainty intervals.

        Args:
            test_dataset: test dataset
            corrected: whether to use Bonferroni-corrected calibration scores

        Returns:
            point predictions and their MAE compared to ground truth
        """
        self.auxiliary_forecaster.eval()

        point_predictions = []
        errors = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            point_prediction, _ = self.auxiliary_forecaster(sequences)
            batch_intervals, _ = self.predict(sequences, corrected=corrected)
            point_predictions.append(point_prediction)
            errors.append(torch.nn.functional.l1_loss(point_prediction, targets, reduction="none").squeeze())

        point_predictions = torch.cat(point_predictions)
        errors = torch.cat(errors)

        return point_predictions, errors


class AdaptiveCFRNN(CFRNN, torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        input_size=1,
        output_size=1,
        horizon=1,
        error_rate=0.05,
        rnn_mode="LSTM",
        auxiliary_forecaster_path=None,
        beta=1,
        **kwargs
    ):
        super(AdaptiveCFRNN, self).__init__(
            embedding_size, input_size, output_size, horizon, error_rate, rnn_mode, auxiliary_forecaster_path
        )

        # Normalisation network
        self.normalising_rnn = torch.nn.RNN(
            input_size=self.input_size, hidden_size=self.embedding_size, batch_first=True
        )

        self.normalising_out = torch.nn.Linear(self.embedding_size, self.horizon * self.output_size)

        self.beta = beta

    def forward(self, sequences):
        """Returns an estimate of normalisation target ln|y - hat{y}|."""
        _, h_n = self.normalising_rnn(sequences.float())
        out = self.normalising_out(h_n).reshape(-1, self.horizon, self.output_size)
        return out

    def score(self, sequences, lengths):
        out = self.forward(sequences)
        return torch.exp(out) + self.beta

    def train_normaliser(self, train_dataset, batch_size, epochs):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            train_loss = 0.0

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                # Get the RNN multi-horizon forecast.
                forecaster_out, _ = self.auxiliary_forecaster(sequences)
                lengths_mask = get_lengths_mask(sequences, lengths, self.horizon)

                # Compute normalisation target ln|y - \hat{y}|.
                normalisation_target = torch.log(torch.abs(targets - forecaster_out)) * lengths_mask

                # Normalising network estimates the normalisation target.
                out = self.forward(sequences)
                loss = criterion(out.float(), normalisation_target.float())
                loss.backward()

                train_loss += loss.item()
                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 100 == 0:
                print("Epoch: {}\tNormalisation loss: {}".format(epoch, mean_train_loss))

    def nonconformity(self, output, calibration_example):
        """Measures the nonconformity between output and target time series."""
        sequence, target, length = calibration_example
        score = torch.nn.functional.l1_loss(output, target, reduction="none")
        normalised_score = score / self.score(sequence, length)
        return normalised_score

    def fit(
        self, train_dataset, calibration_dataset, epochs, lr, normaliser_epochs=500, batch_size=32
    ):  # pylint: disable=arguments-differ
        if self.requires_auxiliary_fit:
            # Train the multi-horizon forecaster.
            self.auxiliary_forecaster.fit(train_dataset, batch_size, epochs, lr)

        # Train normalisation network.
        self.train_normaliser(train_dataset, batch_size, normaliser_epochs)
        self.normalising_rnn.eval()
        self.normalising_out.eval()

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True):
        """Forecasts the time series with conformal uncertainty intervals."""
        out, hidden = self.auxiliary_forecaster(x, state)

        score = self.score(x, len(x))
        if not corrected:
            # [batch_size, horizon, n_outputs]
            lower = out - self.critical_calibration_scores * score
            upper = out + self.critical_calibration_scores * score

        else:
            # [batch_size, horizon, n_outputs]
            lower = out - self.corrected_critical_calibration_scores * score
            upper = out + self.corrected_critical_calibration_scores * score

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden
