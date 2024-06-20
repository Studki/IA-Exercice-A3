# A5_GRU/output_options.py
import numpy as np
from gru import GRU

def gru_sequence_output(model, inputs):
    h = np.zeros((model.hidden_size, 1))
    outputs = []
    for x in inputs:
        h = model.forward(x, h)
        outputs.append(h)
    return outputs

def gru_last_output(model, inputs):
    h = np.zeros((model.hidden_size, 1))
    for x in inputs:
        h = model.forward(x, h)
    return h

def gru_hidden_outputs(model, inputs):
    h = np.zeros((model.hidden_size, 1))
    outputs = []
    for x in inputs:
        h = model.forward(x, h)
        outputs.append(h)
    return outputs
