import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Input, Embedding, Dense, Concatenate


class ConcatenateContext(Layer):
    def __init__(self):
        super(ConcatenateContext, self).__init__()
        self.concat_layer = Concatenate(axis=-1)

    def call(self, inputs):
        context_vector, decoder_outputs = inputs
        expanded_context = tf.expand_dims(context_vector, 1)
        
        # Tile context vector to match decoder_outputs shape
        context_tiled = tf.tile(expanded_context, [1, tf.shape(decoder_outputs)[1], 1])
        
        # Concatenate context vector with decoder outputs
        concatenated = self.concat_layer([context_tiled, decoder_outputs])
        return concatenated