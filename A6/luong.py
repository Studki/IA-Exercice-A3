import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)

    def call(self, query, values):
        score = tf.matmul(query, self.W(values), transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, values)
        return context_vector, attention_weights
