# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

import numpy as np
import torch

from influence.influence_computation import influence_function
from influence.influence_utils import perturb_model_


class RNN_uncertainty_wrapper:
    def __init__(self, model, mode="exact", damp=1e-4):

        self.model = model
        self.IF = influence_function(model, train_index=list(range(model.X.shape[0])), mode=mode, damp=damp)

        X_ = [model.X[k][: int(torch.sum(model.masks[k, :]).detach().numpy())] for k in range(model.X.shape[0])]
        self.LOBO_residuals = []

        for k in range(len(self.IF)):
            perturbed_models = perturb_model_(self.model, self.IF[k])
            self.LOBO_residuals.append(
                np.abs(
                    np.array(self.model.y[k]).reshape(-1, 1) - np.array(perturbed_models.predict(X_[k], padd=False)).T
                )
            )

            del perturbed_models

        self.LOBO_residuals = np.squeeze(np.array(self.LOBO_residuals))

    def predict(self, X_test, coverage=0.95):

        variable_preds = []
        num_sequences = np.array(X_test).shape[0]

        for k in range(len(self.IF)):
            perturbed_models = perturb_model_(self.model, self.IF[k])
            variable_preds.append(perturbed_models.predict(X_test))

            del perturbed_models

        variable_preds = np.array(variable_preds)

        y_u_approx = np.quantile(
            variable_preds + np.repeat(np.expand_dims(self.LOBO_residuals, axis=1), num_sequences, axis=1),
            1 - (1 - coverage) / 2,
            axis=0,
            keepdims=False,
        )
        y_l_approx = np.quantile(
            variable_preds - np.repeat(np.expand_dims(self.LOBO_residuals, axis=1), num_sequences, axis=1),
            (1 - coverage) / 2,
            axis=0,
            keepdims=False,
        )
        y_pred = self.model.predict(X_test)

        return y_pred, y_l_approx, y_u_approx
