import os
import unicodedata
import re

from utils import preprocess_sentence

def create_dataset(path, num_examples):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in line.split('\t')] for line in lines[:num_examples]]
    return zip(*word_pairs)
