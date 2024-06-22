import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def build_model(input_vocab_size, target_vocab_size, max_length_source, max_length_target, embedding_dim, units):
    # Encoder
    encoder_inputs = Input(shape=(max_length_source,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(units, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(max_length_target-1,))
    decoder_embedding = Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True)(decoder_embedding, initial_state=[state_h, state_c])
    decoder_outputs = Dense(target_vocab_size, activation='softmax')(decoder_lstm)

    # Model
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
