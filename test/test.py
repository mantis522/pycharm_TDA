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

RN_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_neg.json"
RP_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_pos.json"
RPPN_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_PP_neg.json"
RRPP_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_PP_pos.json"
RSBARN_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_SBAR_neg.json"
RSBARP_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_SBAR_pos.json"
origin_directory = "C:/Users/ruin/Desktop/data/json_data/train_data_full.json"
test_directory = "C:/Users/ruin/Desktop/data/json_data/test_data_full.json"
RAN = "C:/Users/ruin/Desktop/data/json_data/removed_data/RAN.json"
RAP = "C:/Users/ruin/Desktop/data/json_data/removed_data/RAP.json"


home_origin_directory = "D:/ruin/data/json_data/train_data_full.json"
home_test_directory = "D:/ruin/data/json_data/test_data_full.json"
home_RN_directory = "D:/ruin/data/json_data/removed_data/removed_neg.json"
home_RP_directory = "D:/ruin/data/json_data/removed_data/removed_pos.json"
home_RPPN_directory = "D:/ruin/data/json_data/removed_data/removed_PP_neg.json"
home_RRPP_directory = "D:/ruin/data/json_data/removed_data/removed_PP_pos.json"
home_RSBARN_directory = "D:/ruin/data/json_data/removed_data/removed_SBAR_neg.json"
home_RSBARP_directory = "D:/ruin/data/json_data/removed_data/removed_SBAR_pos.json"

def making_origin_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    train_review = []
    train_label = []

    train_data = json_data['data']

    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])

    df_train = pd.DataFrame(train_review, columns=['data'])
    df_train['label'] = train_label

    return df_train

def making_test_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    test_data = json_data['data']
    test_review = []
    test_label = []

    for a in range(len(test_data)):
        test_review.append(test_data[a]['txt'])
        test_label.append(test_data[a]['label'])

    df = pd.DataFrame(test_review, columns=['data'])
    df['label'] = test_label

    return df

def making_df(file_directory, label):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    removed_sentence = json_data['removed_sentence']
    removed_sentence = removed_sentence[0]
    removed = []

    for a in range(len(removed_sentence)):
        for b in range(len(removed_sentence[a])):
            removed.append(removed_sentence[a][b])

    df = pd.DataFrame(removed)
    df.columns = ['data']
    df['label'] = label

    return df

removed_pos_SBAR = making_df(RSBARN_directory, 0)
removed_neg_SBAR = making_df(RSBARP_directory, 1)

removed_neg_PP = making_df(RPPN_directory, 0)
removed_pos_PP = making_df(RRPP_directory, 1)

removed_neg_JJ = making_df(RN_directory, 0)
removed_pos_JJ = making_df(RP_directory, 1)

Lab_RAN = making_df(RAN, 0)
Lab_RAP = making_df(RAP, 1)

test_df = making_test_df(test_directory)
test_df = test_df.sample(frac=1).reset_index(drop=True)
origin_train_df = making_origin_df(origin_directory)

removed_train_df = pd.concat([Lab_RAN, Lab_RAP])
removed_train_df = removed_train_df.reset_index(drop=True)

concat_train_df = pd.concat([removed_train_df, origin_train_df])
concat_train_df = concat_train_df.reset_index(drop=True)

x_train = concat_train_df['data'].values
y_train = concat_train_df['label'].values

val_df = test_df[:12500]
test_df = test_df[12500:]

print(val_df)
