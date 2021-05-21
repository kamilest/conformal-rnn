import numpy as np
import torch
from scipy import stats as st
from torch import nn
from torch.autograd import Variable

from models.losses import model_loss
from utils.data_padding import padd_arrays, unpadd_arrays

torch.manual_seed(1)


class DPRNN(nn.Module):
    def __init__(self, mode="LSTM", epochs=5, batch_size=150, max_steps=50,
                 input_size=30, lr=0.01, output_size=1, embedding_size=20,
                 n_layers=1, n_steps=50, dropout_prob=0.5, **kwargs):

        super(DPRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.MAX_STEPS = max_steps
        self.INPUT_SIZE = input_size
        self.LR = lr
        self.OUTPUT_SIZE = output_size
        self.HIDDEN_UNITS = embedding_size
        self.NUM_LAYERS = n_layers
        self.N_STEPS = n_steps

        self.mode = mode
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=dropout_prob)

        rnn_dict = {"RNN": nn.RNN(input_size=self.INPUT_SIZE,
                                  hidden_size=self.HIDDEN_UNITS,
                                  num_layers=self.NUM_LAYERS, batch_first=True,
                                  dropout=self.dropout_prob, ),
                    "LSTM": nn.LSTM(input_size=self.INPUT_SIZE,
                                    hidden_size=self.HIDDEN_UNITS,
                                    num_layers=self.NUM_LAYERS,
                                    batch_first=True,
                                    dropout=self.dropout_prob, ),
                    "GRU": nn.GRU(input_size=self.INPUT_SIZE,
                                  hidden_size=self.HIDDEN_UNITS,
                                  num_layers=self.NUM_LAYERS, batch_first=True,
                                  dropout=self.dropout_prob, )
                    }

        self.rnn = rnn_dict[self.mode]
        self.out = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.mode == "LSTM":
            r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero
            # initial hidden state

        else:
            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(self.dropout(h_n))
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
        self.loss_func = model_loss  # nn.MSELoss()

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

                output = self(x).reshape(-1, self.OUTPUT_SIZE)  # rnn output

                loss = self.loss_func(output, y, msk)  # MSE loss

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                if step % 50 == 0:
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)

    def predict(self, X, num_samples=100, alpha=0.05):
        z_critical = st.norm.ppf((1 - alpha) + (alpha) / 2)

        if type(X) is list:
            X_, masks = padd_arrays(X, max_length=self.MAX_STEPS)
        else:
            X_, masks = padd_arrays([X], max_length=self.MAX_STEPS)

        predictions = []
        X_test = Variable(torch.tensor(X_), volatile=True).type(
            torch.FloatTensor)

        for idx in range(num_samples):
            predicts_ = self(X_test).view(-1, self.OUTPUT_SIZE)
            predictions.append(
                predicts_.detach().numpy().reshape((-1, 1, self.OUTPUT_SIZE)))

        pred_mean = unpadd_arrays(
            np.mean(np.concatenate(predictions, axis=1), axis=1), masks)
        pred_std = unpadd_arrays(
            z_critical * np.std(np.concatenate(predictions, axis=1), axis=1),
            masks)

        return pred_mean, pred_std
