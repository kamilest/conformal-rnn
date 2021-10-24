# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

import numpy as np
import torch
from torch.autograd import Variable

from models.losses import quantile_loss
from utils.data_padding import padd_arrays, unpadd_arrays

torch.manual_seed(1)


class QRNN(torch.nn.Module):
    def __init__(self, rnn_mode="LSTM", epochs=5, batch_size=150, max_steps=50,
                 input_size=1, lr=0.01, output_size=1, embedding_size=20,
                 n_layers=1, n_steps=50, alpha=0.05, **kwargs):

        super(QRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.MAX_STEPS = max_steps
        self.INPUT_SIZE = input_size
        self.LR = lr
        self.OUTPUT_SIZE = output_size
        self.HIDDEN_UNITS = embedding_size
        self.NUM_LAYERS = n_layers
        self.N_STEPS = n_steps
        self.q = alpha
        self.rnn_mode = rnn_mode

        rnn_dict = {"RNN": torch.nn.RNN(input_size=self.INPUT_SIZE,
                                        hidden_size=self.HIDDEN_UNITS,
                                        num_layers=self.NUM_LAYERS,
                                        batch_first=True, ),
                    "LSTM": torch.nn.LSTM(input_size=self.INPUT_SIZE,
                                          hidden_size=self.HIDDEN_UNITS,
                                          num_layers=self.NUM_LAYERS,
                                          batch_first=True, ),
                    "GRU": torch.nn.GRU(input_size=self.INPUT_SIZE,
                                        hidden_size=self.HIDDEN_UNITS,
                                        num_layers=self.NUM_LAYERS,
                                        batch_first=True, )
                    }

        self.rnn = rnn_dict[self.rnn_mode]
        self.out = torch.nn.Linear(self.HIDDEN_UNITS, 2 * self.OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.rnn_mode == "LSTM":
            r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero
            # initial hidden state
        else:
            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(h_n)

        return out

    def fit(self, X, Y):
        X_padded, _ = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(
            padd_arrays(Y, max_length=self.OUTPUT_SIZE)[0], axis=2), np.squeeze(
            padd_arrays(Y, max_length=self.OUTPUT_SIZE)[1], axis=2)

        X = Variable(torch.tensor(X_padded), volatile=True).type(
            torch.FloatTensor)
        Y = Variable(torch.tensor(Y_padded), volatile=True).type(
            torch.FloatTensor)
        loss_masks = Variable(torch.tensor(loss_masks), volatile=True).type(
            torch.FloatTensor)

        self.X = X
        self.Y = Y
        self.masks = loss_masks

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.LR)  # optimize all rnn parameters
        self.loss_func = quantile_loss

        # training and testing
        for epoch in range(self.EPOCH):
            for step in range(self.N_STEPS):
                batch_indexes = np.random.choice(list(range(X.shape[0])),
                                                 size=self.BATCH_SIZE,
                                                 replace=True, p=None)

                x = torch.tensor(X[batch_indexes, :, :]).reshape(-1,
                                                                 self.MAX_STEPS,
                                                                 self.INPUT_SIZE).detach()
                y = torch.tensor(Y[batch_indexes]).detach()
                msk = torch.tensor(loss_masks[batch_indexes]).detach()

                output = self(x).reshape(-1, self.OUTPUT_SIZE, 2)  # rnn output

                # quantile loss
                loss = self.loss_func(output[:, :, 0], y, msk,
                                      self.q) + self.loss_func(output[:, :, 1],
                                                               y, msk,
                                                               1 - self.q)

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)

    def predict(self, X):
        if type(X) is list:
            X_, masks = padd_arrays(X, max_length=self.MAX_STEPS)
        else:
            X_, masks = padd_arrays(X, max_length=self.MAX_STEPS)

        X_test = Variable(torch.tensor(X_), volatile=True).type(
            torch.FloatTensor)
        predicts_ = self(X_test).view(-1, self.OUTPUT_SIZE, 2)
        prediction_0 = unpadd_arrays(predicts_[:, :, 0].detach().numpy(), masks)
        prediction_1 = unpadd_arrays(predicts_[:, :, 1].detach().numpy(), masks)

        return prediction_0, prediction_1
