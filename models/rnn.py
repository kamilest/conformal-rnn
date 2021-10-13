# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

import numpy as np
import torch
from torch.autograd import Variable

from models.losses import model_loss, single_losses
from utils.data_padding import padd_arrays, unpadd_arrays


class RNN(torch.nn.Module):
    def __init__(self, rnn_mode="RNN", epochs=5, batch_size=150, max_steps=50,
                 input_size=30, lr=0.01, output_size=1, embedding_size=20,
                 n_layers=1, n_steps=50, **kwargs):

        super(RNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.MAX_STEPS = max_steps
        self.INPUT_SIZE = input_size
        self.LR = lr
        self.OUTPUT_SIZE = output_size
        self.HIDDEN_UNITS = embedding_size
        self.NUM_LAYERS = n_layers
        self.N_STEPS = n_steps

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

        self.rnn_mode = rnn_mode
        self.rnn = rnn_dict[self.rnn_mode]

        self.out = torch.nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

        self.X = None
        self.y = None
        self.masks = None
        self.loss_fn = None
        self.loss = None

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.rnn_mode == "LSTM":
            # None represents zero initial hidden state
            r_out, (h_n, h_c) = self.rnn(x, None)
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

        X = torch.FloatTensor(X_padded)
        Y = torch.FloatTensor(Y_padded)
        loss_masks = torch.FloatTensor(loss_masks)

        self.X = X
        self.y = Y
        self.masks = loss_masks

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.LR)  # optimize all rnn parameters
        self.loss_fn = model_loss  # nn.MSELoss()

        # training and testing
        for epoch in range(self.EPOCH):
            for step in range(self.N_STEPS):
                batch_indexes = np.random.choice(list(range(X.shape[0])),
                                                 size=self.BATCH_SIZE,
                                                 replace=True, p=None)

                # reshape x to (batch, time_step, input_size)
                x = torch.tensor(X[batch_indexes, :, :]).reshape(-1,
                                                                 self.MAX_STEPS,
                                                                 self.INPUT_SIZE).detach()
                y = torch.tensor(Y[batch_indexes]).detach()
                msk = torch.tensor(loss_masks[batch_indexes]).detach()

                output = self(x).reshape(-1, self.OUTPUT_SIZE)  # rnn output

                self.loss = self.loss_fn(output, y, msk)  # MSE loss

                optimizer.zero_grad()  # clear gradients for this training step
                self.loss.backward(
                    retain_graph=True)  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            if epoch % 50 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % self.loss.data)

    def predict(self, X, padd=False):
        if type(X) is list:
            X_, masks = padd_arrays(X, max_length=self.MAX_STEPS)
        else:
            X_, masks = padd_arrays([X], max_length=self.MAX_STEPS)

        X_test = Variable(torch.tensor(X_), volatile=True).type(
            torch.FloatTensor)
        predicts_ = self(X_test).view(-1, self.OUTPUT_SIZE)

        if padd:
            prediction = unpadd_arrays(predicts_.detach().numpy(), masks)
        else:
            prediction = predicts_.detach().numpy()

        return prediction

    def sequence_loss(self):
        return single_losses(self)
