import tensorflow as tf
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bahdanau import BahdanauAttention
from luong import LuongAttention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='bahdanau', rnn_type='lstm'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(self.dec_units)
        else:
            self.attention = LuongAttention(self.dec_units)
        
        if rnn_type == 'lstm':
            self.rnn = tf.keras.layers.LSTM(self.dec_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
        elif rnn_type == 'gru':
            self.rnn = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        # x shape after passing through embedding: (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # context_vector shape: (batch_size, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after concatenation: (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passing the concatenated vector to the RNN
        output, state = self.rnn(x)
        
        # Reshaping output for dense layer: (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # Output shape: (batch_size * 1, vocab_size)
        x = self.fc(output)

        return x, state, attention_weights

