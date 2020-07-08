import json
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import keras
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

RAP_dir = "C:/Users/ruin/Desktop/data/json_data/removed_data/RAP.json"
RAN_dir = "C:/Users/ruin/Desktop/data/json_data/removed_data/RAN.json"

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
test_df = making_test_df(test_directory)

# RAN = making_df(RAN_dir, 0)
# RAP = making_df(RAP_dir, 1)

removed_pos_SBAR = making_df(RSBARN_directory, 0)
removed_neg_SBAR = making_df(RSBARP_directory, 1)

val_df = test_df[:12500]
test_df = test_df[12500:]

removed_train_df = pd.concat([removed_neg_SBAR, removed_pos_SBAR])
removed_train_df = removed_train_df.reset_index(drop=True)

concat_train_df = pd.concat([removed_train_df, origin_train_df])
concat_train_df = concat_train_df.reset_index(drop=True)

x_train = concat_train_df['data'].values
y_train = concat_train_df['label'].values

x_val = val_df['data'].values
y_val = val_df['label'].values

x_test = test_df['data'].values
y_test = test_df['label'].values

vocab_size = 1000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_test)
tokenizer.fit_on_texts(x_val)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=vocab_size)
x_val = sequence.pad_sequences(x_val, maxlen=vocab_size)
x_test = sequence.pad_sequences(x_val, maxlen=vocab_size)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)

model = Sequential()
model.add(Embedding(5000, 120))
model.add(LSTM(120))
model.add(Dense(1, activation='sigmoid'))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val)
                    ,verbose=1)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)

print('acc : ', loss_and_metrics)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# using svg visual model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# spent to time
print("--- %s seconds ---" % (time.time() - start_time))