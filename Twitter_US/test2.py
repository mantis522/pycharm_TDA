import pandas as pd
import numpy as np
import json
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, GRU, BatchNormalization
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
import math

home_test_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_test.csv"
home_train_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_train.csv"

test_dir = "D:/data/csv_file/airline_tweet_test.csv"
train_dir = "D:/data/csv_file/airline_tweet_train.csv"

glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"

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
    df.columns = ['text']
    df['label'] = label

    return df

def original_df(file_directory):
    df = pd.read_csv(file_directory)
    df = df[['airline_sentiment', 'text']]
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    # df.rename(columns={"text": "data"}, inplace=True)

    return df

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

original_train_df = original_df(train_dir)
original_test_df = original_df(test_dir)

# print(original_train_df['text'])
# input_text = "this is a test."
#
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model_gpt = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=gpt_tokenizer.eos_token_id)
#
# input_ids = gpt_tokenizer.encode(input_text, return_tensors='tf')
# cur_len = shape_list(input_ids)[1]
# greedy_output = model_gpt.generate(input_ids, max_length=cur_len + 35)
# output_text = gpt_tokenizer.decode(greedy_output[0], skip_special_tokens=True)
# output_text = " ".join(output_text.split())
# output_text = sent_tokenize(output_text)[1]
# print(output_text)

