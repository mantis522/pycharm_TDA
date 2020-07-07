import json
import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

origin_train_df = making_origin_df(origin_directory)
test_df = making_test_df(test_directory)


origin_train_df = pd.concat([origin_train_df] * 1, ignore_index=True)

X_train = origin_train_df['data'].values
y_train = origin_train_df['label'].values

X_val = test_df['data'].values
y_val = test_df['label'].values

vocab_size = 1000

tokenizer = Tokenizer(num_words=vocab_size)
# tokenizer.fit_on_texts(X_train)

# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_seq = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_val_seq = tokenizer.sequences_to_matrix(X_val, mode='binary')

y_train = keras.utils.to_categorical(y_train, 2)
y_val = keras.utils.to_categorical(y_val, 2)

l = [len(i) for i in X_train_seq]
l = np.array(l)

maxlen = 80

print('minimum number of words: {}'.format(l.min()))
print('median number of words: {}'.format(np.median(l)))
print('average number of words: {}'.format(l.mean()))
print('maximum number of words: {}'.format(l.max()))

X_train_seq = sequence.pad_sequences(X_train_seq, maxlen=maxlen)
X_val_seq = sequence.pad_sequences(X_val_seq, maxlen=maxlen)

print(X_train_seq.shape)
print(X_val_seq.shape)
print(y_train)
print(y_train.shape)

print('Build model...')
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=vocab_size))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# model.add(Embedding(vocab_size, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
# hist = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), nb_epoch=2, batch_size=64, verbose=1)
hist = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), batch_size=64, epochs=10, verbose=1)
score, acc = model.evaluate(X_val_seq, y_val, batch_size=64, verbose=0)
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