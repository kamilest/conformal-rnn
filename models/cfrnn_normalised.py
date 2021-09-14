# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import torch
from cfrnn import CFRNN, get_critical_scores


class CFRNN_normalised(CFRNN):
    def __init__(self, beta=1, **kwargs):
        super(CFRNN, self).__init__()

        # Normalisation network
        self.normalising_rnn = torch.nn.RNN(input_size=self.input_size,
                                            hidden_size=self.embedding_size,
                                            batch_first=True)

        self.normalising_out = torch.nn.Linear(self.embedding_size,
                                               self.horizon * self.output_size)

        self.beta = beta

        self.normalised_calibration_scores = None
        self.normalised_critical_calibration_scores = None
        self.normalised_corrected_critical_calibration_scores = None

    def normaliser_forward(self, sequences):
        """Returns an estimate of normalisation target ln|y - hat{y}|."""
        _, h_n = self.normalising_rnn(sequences.float())
        out = self.normalising_out(h_n).reshape(-1, self.horizon,
                                                self.output_size)
        return out

    def normalisation_score(self, sequences, lengths):
        out = self.normaliser_forward(sequences)
        return torch.exp(out) + self.beta

    def train_normaliser(self, train_loader):
        # TODO tuning
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # TODO early stopping based on validation loss of the calibration set
        for epoch in range(500):
            train_loss = 0.

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                # Get the RNN multi-horizon forecast.
                forecaster_out, _ = self(sequences)
                lengths_mask = self.get_lengths_mask(sequences, lengths)

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
                batch_size=32):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)

        # Train the multi-horizon forecaster.
        self.train_forecaster(train_loader, epochs, lr)

        # Train normalisation network.
        self.train_normaliser(train_loader)
        self.normalising_rnn.eval()
        self.normalising_out.eval()

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True, normalised=False):
        """Forecasts the time series with conformal uncertainty intervals."""
        out, hidden = self(x, state)

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