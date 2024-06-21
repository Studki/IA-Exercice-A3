import tensorflow as tf
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bahdanau import BahdanauAttention
from luong import LuongAttention

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if rnn_type == 'lstm':
            self.rnn = tf.keras.layers.LSTM(self.enc_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
        elif rnn_type == 'gru':
            self.rnn = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        if isinstance(self.rnn, tf.keras.layers.LSTM):
            output, state_h, state_c = self.rnn(x, initial_state=hidden)
            state = [state_h, state_c]
        else:  # GRU
            output, state = self.rnn(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        if isinstance(self.rnn, tf.keras.layers.LSTM):
            return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]
        return tf.zeros((self.batch_sz, self.enc_units))