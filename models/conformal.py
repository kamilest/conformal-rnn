import torch


class ConformalForecaster:
    def __init__(self, **kwargs):
        super(ConformalForecaster, self).__init__()

        # Encoder and forecaster can be the same (if embeddings are
        # trained on `horizon`-step forecasts), but different models are
        # possible.
        # TODO separate encoder and forecaster models.
        # TODO try the RNN autoencoder trained on reconstruction error.
        self.encoder = None
        self.forecaster = self.encoder

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
