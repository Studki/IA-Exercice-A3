import numpy as np

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weights for the input gate, forget gate, cell gate, and output gate
        self.W_i = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim))
        self.U_i = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.b_i = np.zeros((hidden_dim,))

        self.W_f = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim))
        self.U_f = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.b_f = np.zeros((hidden_dim,))

        self.W_o = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim))
        self.U_o = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.b_o = np.zeros((hidden_dim,))

        self.W_c = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim))
        self.U_c = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.b_c = np.zeros((hidden_dim,))

    def forward(self, x, h_prev, C_prev):
        # Input gate
        i_t = self._sigmoid(np.dot(self.W_i, x) + np.dot(self.U_i, h_prev) + self.b_i)
        # Forget gate
        f_t = self._sigmoid(np.dot(self.W_f, x) + np.dot(self.U_f, h_prev) + self.b_f)
        # Output gate
        o_t = self._sigmoid(np.dot(self.W_o, x) + np.dot(self.U_o, h_prev) + self.b_o)
        # Cell gate
        C_hat_t = np.tanh(np.dot(self.W_c, x) + np.dot(self.U_c, h_prev) + self.b_c)
        # New cell state
        C_t = f_t * C_prev + i_t * C_hat_t
        # New hidden state
        h_t = o_t * np.tanh(C_t)

        self.h_t, self.C_t, self.i_t, self.f_t, self.o_t, self.C_hat_t = h_t, C_t, i_t, f_t, o_t, C_hat_t

        return h_t, C_t

    def backward(self, x, h_prev, C_prev, dh_next, dC_next):
        # Gradient of output gate
        do_t = dh_next * np.tanh(self.C_t)
        dWo = np.outer(do_t * self.o_t * (1 - self.o_t), x)
        dUo = np.outer(do_t * self.o_t * (1 - self.o_t), h_prev)
        dbo = do_t * self.o_t * (1 - self.o_t)

        # Gradient of cell state
        dC_t = dC_next + dh_next * self.o_t * (1 - np.tanh(self.C_t)**2)

        # Gradient of cell gate
        dC_hat_t = dC_t * self.i_t
        dWc = np.outer(dC_hat_t * (1 - self.C_hat_t**2), x)
        dUc = np.outer(dC_hat_t * (1 - self.C_hat_t**2), h_prev)
        dbc = dC_hat_t * (1 - self.C_hat_t**2)

        # Gradient of input gate
        di_t = dC_t * self.C_hat_t
        dWi = np.outer(di_t * self.i_t * (1 - self.i_t), x)
        dUi = np.outer(di_t * self.i_t * (1 - self.i_t), h_prev)
        dbi = di_t * self.i_t * (1 - self.i_t)

        # Gradient of forget gate
        df_t = dC_t * C_prev
        dWf = np.outer(df_t * self.f_t * (1 - self.f_t), x)
        dUf = np.outer(df_t * self.f_t * (1 - self.f_t), h_prev)
        dbf = df_t * self.f_t * (1 - self.f_t)

        # Aggregate gradients
        dW = dWi + dWf + dWo + dWc
        dU = dUi + dUf + dUo + dUc
        db = dbi + dbf + dbo + dbc

        dh_prev = np.dot(self.U_i.T, di_t) + np.dot(self.U_f.T, df_t) + np.dot(self.U_o.T, do_t) + np.dot(self.U_c.T, dC_hat_t)
        dC_prev = dC_t * self.f_t

        return dW, dU, db, dh_prev, dC_prev

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))