import json
import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
from matplotlib import pyplot as plt

start_time = time.time()

## label 1은 부정 / 2가 긍정
data_train = pd.read_csv("D:/data/amazon_review_polarity_csv/train.csv", names=['label', 'title', 'review'])
data_train.loc[data_train["label"] == 1, "label"] = 0
data_train.loc[data_train['label'] == 2, "label"] = 1
data_train.drop(['title'], axis='columns', inplace=True)
print(len(data_train))

data_test = pd.read_csv("D:/data/amazon_review_polarity_csv/test.csv", names=['label', 'title', 'review'])
data_test.loc[data_test["label"] == 1, "label"] = 0
data_test.loc[data_test["label"] == 2, "label"] = 1
data_test.drop(['title'], axis='columns', inplace=True)

print(len(data_test))