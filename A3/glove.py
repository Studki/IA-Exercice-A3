import numpy as np

class GloVe:
    def __init__(self, vocab_size, embed_size, alpha=0.75, x_max=100):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.alpha = alpha
        self.x_max = x_max
        self.W = np.random.uniform(-1, 1, (vocab_size, embed_size))
        self.b = np.random.uniform(-1, 1, vocab_size)
        self.grad_squared_W = np.ones((vocab_size, embed_size))
        self.grad_squared_b = np.ones(vocab_size)

    def cooccurrence_matrix(self, words, vocab, window_size=2):
        matrix = np.zeros((len(vocab), len(vocab)))
        for i, word in enumerate(words):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    matrix[vocab[word], vocab[words[j]]] += 1
        return matrix

    def train(self, words, vocab, epochs, learning_rate):
        cooccurrences = self.cooccurrence_matrix(words, vocab)
        for epoch in range(epochs):
            total_cost = 0
            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    if cooccurrences[i, j] > 0:
                        weight = (cooccurrences[i, j] / self.x_max)**self.alpha if cooccurrences[i, j] < self.x_max else 1
                        cost = weight * (np.dot(self.W[i], self.W[j]) + self.b[i] + self.b[j] - np.log(cooccurrences[i, j]))**2
                        total_cost += cost

                        grad_Wi = weight * (np.dot(self.W[i], self.W[j]) + self.b[i] + self.b[j] - np.log(cooccurrences[i, j])) * self.W[j]
                        grad_Wj = weight * (np.dot(self.W[i], self.W[j]) + self.b[i] + self.b[j] - np.log(cooccurrences[i, j])) * self.W[i]
                        grad_bi = weight * (np.dot(self.W[i], self.W[j]) + self.b[i] + self.b[j] - np.log(cooccurrences[i, j]))
                        grad_bj = grad_bi

                        self.grad_squared_W[i] += grad_Wi**2
                        self.grad_squared_W[j] += grad_Wj**2
                        self.grad_squared_b[i] += grad_bi**2
                        self.grad_squared_b[j] += grad_bj**2

                        self.W[i] -= learning_rate * grad_Wi / np.sqrt(self.grad_squared_W[i])
                        self.W[j] -= learning_rate * grad_Wj / np.sqrt(self.grad_squared_W[j])
                        self.b[i] -= learning_rate * grad_bi / np.sqrt(self.grad_squared_b[i])
                        self.b[j] -= learning_rate * grad_bj / np.sqrt(self.grad_squared_b[j])

            print(f'Epoch: {epoch + 1}, Cost: {total_cost / np.sum(cooccurrences)}')
