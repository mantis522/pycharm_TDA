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

# origin_neg_directory = "C:/Users/ruin/Desktop/data/json_data/train_neg_full.json"
# origin_pos_directory = "C:/Users/ruin/Desktop/data/json_data/train_pos_full.json"
origin_directory = "C:/Users/ruin/Desktop/data/train_data_full.json"
test_directory = "C:/Users/ruin/Desktop/data/json_data/test_data_full.json"

home_origin_dir = "D:/ruin/data/json_data/train_data_full.json"
home_test_dir = "D:/ruin/data/json_data/test_data_full.json"
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

removed_pos_SBAR = making_df(home_RN_directory, 0)
removed_neg_SBAR = making_df(home_RP_directory, 1)
test_df = making_test_df(home_test_dir)

origin_train_df = making_origin_df(home_origin_dir)

removed_train_df = pd.concat([removed_pos_SBAR, removed_neg_SBAR])
removed_train_df = removed_train_df.reset_index(drop=True)

concat_train_df = pd.concat([removed_train_df, origin_train_df])
concat_train_df = concat_train_df.reset_index(drop=True)

x_train = concat_train_df['data'].values
y_train = concat_train_df['label'].values

val_df = test_df[:12500]
test_df = test_df[12500:]

x_val = val_df['data'].values
y_val = val_df['label'].values

x_test = test_df['data'].values
y_test = test_df['label'].values

vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

maxlen = 80


x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print(x_train.shape)
print(x_val.shape)
print(y_train)
print(y_train.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=5, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('Test score : ', score)
print('Test accuracy : ', acc)

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