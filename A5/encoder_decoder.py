# A5_GRU/encoder_decoder.py
import numpy as np
from gru import GRU

class Encoder:
    def __init__(self, model):
        self.model = model

    def encode(self, inputs):
        h = np.zeros((self.model.hidden_size, 1))
        for x in inputs:
            h = self.model.forward(x, h)
        return h

class Decoder:
    def __init__(self, model):
        self.model = model

    def decode(self, inputs, h):
        outputs = []
        for x in inputs:
            h = self.model.forward(x, h)
            outputs.append(h)
        return outputs
