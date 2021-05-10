import numpy as np
import torch
from torch import nn




class ConformalForecaster(nn.Module):
    def __init__(self, embedding_size, input_size=1, horizon=1, alpha=0.05):
        super(ConformalForecaster, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.

        # Encoder and forecaster can be the same (if embeddings are
        # trained on `horizon`-step forecasts), but different models are
        # possible.

        # TODO try separate encoder and forecaster models.
        # TODO try the RNN autoencoder trained on reconstruction error.
        self.encoder = None

        # Single-shot multi-output univariate time series forecaster.
        # https://www.tensorflow.org/tutorials/structured_data/time_series#rnn_2
        # TODO consider autoregressive multi-output model:
        # https://www.tensorflow.org/tutorials/structured_data/time_series#advanced_autoregressive_model
        self.forecaster_rnn = nn.LSTM(input_size=input_size,
                                      hidden_size=embedding_size,
                                      batch_first=True)
        self.forecaster_out = nn.Linear(embedding_size, horizon)

        self.num_train = None
        self.calibration_scores = None
        self.masks = None  # indicates where which parts of time series are
        # padded.

    def fit(self, X, Y):
        # Train encoder to generate embeddings

        # Train forecaster to give forecasts based on embeddings

        # encoder and forecaster can be the same model.

        self.calibrate(None, None)
        pass

    def get_nonconformity(self, output, target):
        """Measures the nonconformity between output and target time series."""

        # Average MSE loss for every step in the sequence.
        # TODO alternative nonconformity scores.
        return torch.mean((self.masks * (output - target)) ** 2, dim=0)

    def calibrate(self, X_cal, Y_cal):
        """
        Computes the nonconformity scores for the calibration set.
        X_cal and Y_cal are expected to be tensors.
        """
        Y_pred = self.forecaster(X_cal)
        self.calibration_scores = torch.sort(self.get_nonconformity(Y_pred,
                                                                    Y_cal))

    def predict(self, X):
        """Forecasts the time series with conformal uncertainty intervals."""
        # TODO implementation.
        pass
