import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from models.losses import model_loss, single_losses
from utils.data_padding import padd_arrays, unpadd_arrays


class RNN(nn.Module):

    def __init__(self,
                 mode="RNN",
                 EPOCH=5,
                 BATCH_SIZE=150,
                 MAX_STEPS=50,
                 INPUT_SIZE=30,
                 LR=0.01,
                 OUTPUT_SIZE=1,
                 HIDDEN_UNITS=20,
                 NUM_LAYERS=1,
                 N_STEPS=50):

        super(RNN, self).__init__()

        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_STEPS = MAX_STEPS
        self.INPUT_SIZE = INPUT_SIZE
        self.LR = LR
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.HIDDEN_UNITS = HIDDEN_UNITS
        self.NUM_LAYERS = NUM_LAYERS
        self.N_STEPS = N_STEPS

        rnn_dict = {"RNN": nn.RNN(input_size=self.INPUT_SIZE,
                                  hidden_size=self.HIDDEN_UNITS,
                                  num_layers=self.NUM_LAYERS,
                                  batch_first=True, ),
                    "LSTM": nn.LSTM(input_size=self.INPUT_SIZE,
                                    hidden_size=self.HIDDEN_UNITS,
                                    num_layers=self.NUM_LAYERS,
                                    batch_first=True, ),
                    "GRU": nn.GRU(input_size=self.INPUT_SIZE,
                                  hidden_size=self.HIDDEN_UNITS,
                                  num_layers=self.NUM_LAYERS,
                                  batch_first=True, )
                    }

        self.mode = mode
        self.rnn = rnn_dict[self.mode]

        self.out = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.mode == "LSTM":

            r_out, (h_n, h_c) = self.rnn(x,
                                         None)  # None represents zero
            # initial hidden state

        else:

            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, :, :])

        return out

    def fit(self, X, Y):
        X_padded, _ = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(
            padd_arrays(Y, max_length=self.MAX_STEPS)[0], axis=2), np.squeeze(
            padd_arrays(Y, max_length=self.MAX_STEPS)[1], axis=2)

        X = Variable(torch.tensor(X_padded), volatile=True).type(
            torch.FloatTensor)
        Y = Variable(torch.tensor(Y_padded), volatile=True).type(
            torch.FloatTensor)
        loss_masks = Variable(torch.tensor(loss_masks), volatile=True).type(
            torch.FloatTensor)

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

                x = torch.tensor(X[batch_indexes, :, :])
                y = torch.tensor(Y[batch_indexes])
                msk = torch.tensor(loss_masks[batch_indexes])

                b_x = Variable(x.view(-1, self.MAX_STEPS,
                                      self.INPUT_SIZE))  # reshape x to (
                # batch, time_step, input_size)
                b_y = Variable(y)  # batch y
                b_m = Variable(msk)

                output = self(b_x).view(-1, self.MAX_STEPS)  # rnn output

                self.loss = self.loss_fn(output, b_y, b_m)  # MSE loss

                optimizer.zero_grad()  # clear gradients for this training step
                self.loss.backward(
                    retain_graph=True)  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                if step % 50 == 0:
                    print('Epoch: ', epoch,
                          '| train loss: %.4f' % self.loss.data)

    def predict(self, X, padd=False):
        if type(X) is list:
            X_, masks = padd_arrays(X, max_length=self.MAX_STEPS)
        else:
            X_, masks = padd_arrays([X], max_length=self.MAX_STEPS)

        X_test = Variable(torch.tensor(X_), volatile=True).type(
            torch.FloatTensor)
        predicts_ = self(X_test).view(-1, self.MAX_STEPS)

        if padd:
            prediction = unpadd_arrays(predicts_.detach().numpy(), masks)
        else:
            prediction = predicts_.detach().numpy()

        return prediction

    def sequence_loss(self):
        return single_losses(self)