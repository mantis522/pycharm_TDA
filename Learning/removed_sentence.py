import json
import pandas as pd
from tqdm.notebook import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np

RN_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_neg.json"
RP_directory = "C:/Users/ruin/Desktop/data/json_data/removed_data/removed_pos.json"
origin_directory = "C:/Users/ruin/Desktop/data/json_data/train_data_full.json"
test_directory = "C:/Users/ruin/Desktop/data/json_data/test_data_full.json"

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

origin_train_df = making_origin_df(origin_directory)
removed_neg = making_df(RN_directory, 0)
removed_pos = making_df(RP_directory, 1)
test_df = making_test_df(test_directory)

removed_train_df = pd.concat([removed_neg, removed_pos])
removed_train_df = removed_train_df.reset_index(drop=True)

concat_train_df = pd.concat([removed_train_df, origin_train_df])
concat_train_df = concat_train_df.reset_index(drop=True)

review_data = concat_train_df['data'].tolist()
review_label = concat_train_df['label'].tolist()
test_data = test_df['data'].tolist()
test_label = test_df['label'].tolist()

max_features = 500

tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(review_data)
X_train = tokenizer.texts_to_sequences(review_data)
X_train = sequence.pad_sequences(X_train, maxlen=max_features)
Y_train = review_label

tokenizer.fit_on_texts(test_data)
X_test = tokenizer.texts_to_sequences(test_data)
X_test = sequence.pad_sequences(X_test, maxlen=max_features)
Y_test = test_label

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

model = Sequential()
model.add(Embedding(5000, 120))
model.add(LSTM(120))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64, callbacks=[es, mc])