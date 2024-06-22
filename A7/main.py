import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from models import build_model
from data_processing import create_dataset

# Constants
embedding_dim = 256
units = 102
BATCH_SIZE = 64
EPOCHS = 10
max_length_source = 10  # Example max length of source sequence
max_length_target = 17

# Paths and dataset
path_to_zip = tf.keras.utils.get_file('fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip', extract=True)
path_to_file = os.path.dirname(path_to_zip) + "/fra.txt"

# Create dataset
num_examples = 30000  # Adjust this based on your dataset size
source_sentences, target_sentences = create_dataset(path_to_file, num_examples)

# Tokenizer for source (English) sentences
input_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(source_sentences)
input_sequences = input_tokenizer.texts_to_sequences(source_sentences)
input_maxlen = max(len(seq) for seq in input_sequences)
input_vocab_size = len(input_tokenizer.word_index) + 1

# Tokenizer for target (French) sentences
target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_sentences)
target_sequences = target_tokenizer.texts_to_sequences(target_sentences)
target_maxlen = max(len(seq) for seq in target_sequences)
target_vocab_size = len(target_tokenizer.word_index) + 1

# Padding sequences to uniform length
input_sequences_padded = pad_sequences(input_sequences, maxlen=input_maxlen, padding='post')
target_sequences_padded = pad_sequences(target_sequences, maxlen=target_maxlen, padding='post')

# Build and compile the model
model = build_model(input_vocab_size, target_vocab_size, max_length_source, max_length_target, embedding_dim, units)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare decoder inputs (shifted)
decoder_inputs_shifted = target_sequences_padded[:, :-1]

# Prepare target sequences (shifted)
target_sequences_shifted = target_sequences_padded[:, 1:]

# Example shape check
print("Shape of decoder_inputs_shifted:", decoder_inputs_shifted.shape)
print("Shape of target_sequences_shifted:", target_sequences_shifted.shape)

# Train the model
model.fit([input_sequences_padded, decoder_inputs_shifted],
          target_sequences_shifted,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

# Save the model if needed
# model.save('translation_model.h5')

# Test the model on new data (optional)
# Replace with your own test sentences and use the trained model to translate
# test_source_sentences = ['Your test sentences here']
# test_source_sequences = input_tokenizer.texts_to_sequences(test_source_sentences)
# test_source_sequences_padded = pad_sequences(test_source_sequences, maxlen=input_maxlen, padding='post')
# predicted_sequences = model.predict(test_source_sequences_padded)
# Decode predicted_sequences to get translated sentences

# Evaluate the model (optional)
# Implement evaluation metrics such as BLEU score to evaluate translation quality
