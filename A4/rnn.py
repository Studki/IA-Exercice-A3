from datetime import datetime
import sys
from layer import RNNLayer
from output import Softmax
from LSTM import LSTMCell
from datetime import datetime
import numpy as np
import sys
from output import Softmax

class Model:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.lstm = LSTMCell(word_dim, hidden_dim)
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        h = np.zeros((T + 1, self.hidden_dim))
        C = np.zeros((T + 1, self.hidden_dim))
        o = np.zeros((T, self.word_dim))

        for t in range(T):
            x_onehot = np.zeros(self.word_dim)
            x_onehot[x[t]] = 1
            h[t + 1], C[t + 1] = self.lstm.forward(x_onehot, h[t], C[t])
            o[t] = np.dot(self.V, h[t + 1])

        return o, h, C

    def predict(self, x):
        o, _, _ = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, x, y):
        output = Softmax()
        o, _, _ = self.forward_propagation(x)
        loss = 0.0
        for t in range(len(y)):
            loss += output.loss(o[t], y[t])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y):
        T = len(y)
        o, h, C = self.forward_propagation(x)
        dU, dW, dV = np.zeros_like(self.lstm.U_i), np.zeros_like(self.lstm.W_i), np.zeros_like(self.V)
        db = np.zeros(self.hidden_dim)
        dh_next = np.zeros(self.hidden_dim)
        dC_next = np.zeros(self.hidden_dim)

        output = Softmax()
        for t in reversed(range(T)):
            dV += np.outer(output.diff(o[t], y[t]), h[t + 1])
            dh = np.dot(self.V.T, output.diff(o[t], y[t])) + dh_next
            dW_t, dU_t, db_t, dh_next, dC_next = self.lstm.backward(
                np.eye(self.word_dim)[x[t]], h[t], C[t], dh, dC_next)
            dW += dW_t
            dU += dU_t
            db += db_t

        return dU, dW, dV, db

    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV, db = self.bptt(x, y)
        for attr in ['i', 'f', 'o', 'c']:
            setattr(self.lstm, 'W_' + attr, getattr(self.lstm, 'W_' + attr) - learning_rate * dW)
            setattr(self.lstm, 'U_' + attr, getattr(self.lstm, 'U_' + attr) - learning_rate * dU)
            setattr(self.lstm, 'b_' + attr, getattr(self.lstm, 'b_' + attr) - learning_rate * db)
        self.V -= learning_rate * dV

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        num_examples_seen = 0
        losses = []
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            for i in range(len(Y)):
                self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
        return losses