import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W1 = np.random.uniform(-1, 1, (vocab_size, embed_size))
        self.W2 = np.random.uniform(-1, 1, (embed_size, vocab_size))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def forward(self, X):
        h = np.dot(self.W1.T, X)
        u = np.dot(self.W2.T, h)
        y_pred = self.softmax(u)
        return y_pred, h, u

    def backward(self, X, y, y_pred, h, learning_rate):
        e = y_pred - y
        dW2 = np.outer(h, e)
        dW1 = np.outer(X, np.dot(self.W2, e.T))

        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            loss = 0
            for target, context in training_data:
                X = np.zeros(self.vocab_size)
                X[target] = 1
                y = np.zeros(self.vocab_size)
                y[context] = 1

                y_pred, h, u = self.forward(X)
                self.backward(X, y, y_pred, h, learning_rate)

                loss -= np.log(y_pred[context])

            print(f'Epoch: {epoch + 1}, Loss: {loss / len(training_data)}')