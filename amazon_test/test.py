import pandas as pd
import numpy as np
import json
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

train_dir = "D:/data/csv_file/amazon/amazon_1000.csv"
test_dir = "D:/data/csv_file/amazon/amazon_test.csv"
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"

original_train_df = pd.read_csv(train_dir)
original_test_df = pd.read_csv(test_dir)

x_train = original_train_df['review'].values
y_train = original_train_df['label'].values
y_train = to_categorical(np.asarray(y_train))

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)

x_val = original_val_df['review'].values
y_val = original_val_df['label'].values
y_val = to_categorical(np.asarray(y_val))

x_test = original_test_df['review'].values
y_test = original_test_df['label'].values
y_test = to_categorical(np.asarray(y_test))

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

print('X_train size:', x_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', x_test.shape)
print('y_test size:', y_test.shape)
print('X_val size: ', x_val.shape)
print('y_val size: ', y_val.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('Test score : ', score)
print('Test accuracy : ', acc)