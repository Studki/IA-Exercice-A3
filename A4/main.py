import numpy as np
from rnn import Model
import os
from collections import defaultdict

def build_vocab(data_folder):
    word_counts = defaultdict(int)
    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), 'r') as file:
            for line in file:
                for word in line.strip().split():
                    word_counts[word] += 1
    word_to_index = {word: i for i, word in enumerate(word_counts.keys(), 1)}
    word_to_index['<PAD>'] = 0  # Padding token
    return word_to_index

def load_data(data_folder, word_to_index):
    X = []
    Y = []
    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), 'r') as file:
            for line in file:
                sequence = [word_to_index[word] for word in line.strip().split() if word in word_to_index]
                X.append(sequence[:-1])
                Y.append(sequence[1:])
    return X, Y

# Build vocabulary
data_folder = 'data'
word_to_index = build_vocab(data_folder)
vocab_size = len(word_to_index)

# Load data
X, Y = load_data(data_folder, word_to_index)

# Convert lists to numpy arrays for efficient processing
X = [np.array(seq) for seq in X]
Y = [np.array(seq) for seq in Y]

# Define your vocabulary size (word_dim) and hidden layer size (hidden_dim)
word_dim = vocab_size  # Number of unique words in the vocabulary
hidden_dim = 100  # Example value

# Initialize the model
model = Model(word_dim, hidden_dim)

# Train the model
learning_rate = 0.005
nepoch = 100
evaluate_loss_after = 1

losses = model.train(X, Y, learning_rate=learning_rate, nepoch=nepoch, evaluate_loss_after=evaluate_loss_after)

# Save model weights
np.save('model_weights_U_i.npy', model.lstm.U_i)