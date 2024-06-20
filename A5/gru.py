# A5_GRU/gru.py
import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size)

        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev):
        concat = np.vstack((h_prev, x))

        zt = self.sigmoid(self.Wz @ concat + self.bz)
        rt = self.sigmoid(self.Wr @ concat + self.br)
        h_tilde = self.tanh(self.Wh @ np.vstack((rt * h_prev, x)) + self.bh)

        ht = (1 - zt) * h_prev + zt * h_tilde
        return ht
