import tensorflow as tf
import numpy as np
import re
import os
import io
import unicodedata
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from encoder import Encoder
from decoder import Decoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from concatenare import ConcatenateContext
from tensorflow.keras.layers import Reshape
from bahdanau import BahdanauAttention  # Assuming you have a separate file for attention mechanism
from tensorflow.keras.layers import RepeatVector  # Import the RepeatVector layer

embedding_dim = 256
units = 102
BATCH_SIZE = 64
EPOCHS = 10
max_length_source = 10  # Example max length of source sequence
max_length_target = 17

# Download and prepare the dataset
path_to_zip = tf.keras.utils.get_file('fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip', extract=True)
path_to_file = os.path.dirname(path_to_zip)+"/fra.txt"

# Function to preprocess sentences
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in line.split('\t')] for line in lines[:num_examples]]
    return zip(*word_pairs)

# Example: Read and preprocess the entire dataset
num_examples = 30000  # Adjust this based on your dataset size
source_sentences, target_sentences = create_dataset(path_to_file, num_examples)

print("Example Source Sentence:", source_sentences[0])
print("Example Target Sentence:", target_sentences[0])

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

print("Input vocabulary size:", input_vocab_size)
print("Target vocabulary size:", target_vocab_size)
print("Max input sequence length:", input_maxlen)
print("Max target sequence length:", target_maxlen)

# Encoder setup
encoder_inputs = Input(shape=(max_length_source,))
encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder setup
decoder_inputs = Input(shape=(max_length_target - 1,))
decoder_embedding = Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention layer
attention_layer = BahdanauAttention(units)
context_vector, attention_weights = attention_layer(state_h, encoder_outputs)

# Concatenate context vector with decoder outputs after repeating it
context_vector_repeated = RepeatVector(max_length_target - 1)(context_vector)
decoder_combined_context = Concatenate(axis=-1)([context_vector_repeated, decoder_outputs])

# Dense layer for prediction
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare decoder inputs (shifted)
decoder_inputs_shifted = target_sequences_padded[:, :-1]

# Prepare target sequences (shifted)
target_sequences_shifted = target_sequences_padded[:, 1:]

# Example shape check
print("Shape of decoder_inputs_shifted:", decoder_inputs_shifted.shape)
print("Shape of target_sequences_shifted:", target_sequences_shifted.shape)

# Example training
model.fit([input_sequences_padded, decoder_inputs_shifted],  # Use shifted inputs
          target_sequences_shifted,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)