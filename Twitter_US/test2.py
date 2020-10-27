import os
import numpy as np
from tqdm import tqdm
from keras_bert import Tokenizer
import pandas as pd

path = "D:/data/20news-18828"
tagset = [(x, i) for i, x in enumerate(os.listdir(path))]
id_to_labels = {id_: label for label, id_ in tagset}

SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5

token_dict = {}

tokenizer = Tokenizer(token_dict)

def load_data(path, tagset):
    global tokenizer
    indices, labels = [], []
    for folder, label in tagset:
        folder = os.path.join(path, folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name), 'r', encoding="utf-8", errors='ignore') as reader:
                text = reader.read()
                # print(text)
            ids = tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)
            labels.append(label)

    print(len(indices))
    print(len(labels))


print(load_data(path, tagset))