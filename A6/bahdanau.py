import tensorflow as tf
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_length, hidden_size)

        # Expand dimensions to perform addition
        query_with_time_axis = tf.expand_dims(query, 1)

        # Attention score calculation
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # Attention weights calculation
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector calculation
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights