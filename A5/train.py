# A5_GRU/train.py
import numpy as np
from gru import GRU
from output_options import gru_sequence_output, gru_last_output, gru_hidden_outputs
from encoder_decoder import Encoder, Decoder

def generate_data(seq_length, num_sequences, input_size):
    return [np.random.randn(input_size, 1) for _ in range(num_sequences * seq_length)]

def train_model(model, data, output_func):
    outputs = output_func(model, data)
    return outputs

input_size = 10
hidden_size = 20
seq_length = 5
num_sequences = 100

data = generate_data(seq_length, num_sequences, input_size)

gru_model = GRU(input_size, hidden_size)

gru_outputs = train_model(gru_model, data, gru_sequence_output)

print("GRU Outputs: ", gru_outputs)
