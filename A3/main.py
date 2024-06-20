import numpy as np
import re
import string
import requests
import zipfile
from io import BytesIO
from word2vec import Word2Vec
from glove import GloVe
import random


word2vec_epoch = 100
glove_epoch = 100

number_of_random_words = 3000 # Number of random words (so the max, if two words get selected then you get 3000 - 1 etc) to select from the data

# Download and extract Text8 dataset
def download_text8():
    url = 'http://mattmahoney.net/dc/text8.zip'
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall()

# Preprocessing the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    words = text.split()
    return words

# Building vocabulary
def build_vocab(words):
    vocab = {}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

# Generating training data for skip-gram
def generate_training_data(words, vocab, window_size=2):
    training_data = []
    for i, word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                training_data.append((vocab[word], vocab[words[j]]))
    return np.array(training_data)

# Download the dataset
download_text8()

# Load the dataset
with open('text8', 'r') as file:
    text8 = file.read()

# Preprocess the text
words = preprocess_text(text8)

random_subset = random.sample(words, number_of_random_words)

vocab = build_vocab(random_subset)
training_data = generate_training_data(random_subset, vocab)

# Word2Vec
vocab_size = len(vocab)
embed_size = 10  # Typically larger for real datasets
word2vec = Word2Vec(vocab_size, embed_size)

word2vec.train(training_data, epochs=word2vec_epoch, learning_rate=0.10)  # Adjust epochs and learning rate as needed

# Getting word embeddings
word_embeddings_w2v = word2vec.W1
print("Word2Vec Embeddings shape:", word_embeddings_w2v.shape)

# GloVe
glove = GloVe(vocab_size, embed_size)
glove.train(random_subset, vocab, epochs=glove_epoch, learning_rate=0.10)  # Adjust epochs and learning rate as needed

# Getting word embeddings
word_embeddings_glove = glove.W
print("GloVe Embeddings shape:", word_embeddings_glove.shape)

# Example similarity calculation for Word2Vec
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

word1 = random.choice(random_subset)
word2 = random.choice(random_subset)

if word1 in vocab and word2 in vocab:
    vec1 = word2vec.W1[vocab[word1]]
    vec2 = word2vec.W1[vocab[word2]]

    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine Similarity between '{word1}' and '{word2}' (Word2Vec):", similarity)

    # Example similarity calculation for GloVe
    vec1_glove = glove.W[vocab[word1]]
    vec2_glove = glove.W[vocab[word2]]

    similarity_glove = cosine_similarity(vec1_glove, vec2_glove)
    print(f"Cosine Similarity between '{word1}' and '{word2}' (GloVe):", similarity_glove)
else:
    print(f"Words '{word1}' or '{word2}' not in vocabulary")
