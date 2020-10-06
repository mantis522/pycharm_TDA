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

data_test = pd.read_csv("D:/data/amazon_review_polarity_csv/test.csv", names=['label', 'title', 'review'])
data_test.loc[data_test["label"] == 1, "label"] = 0
data_test.loc[data_test["label"] == 2, "label"] = 1
data_test.drop(['title'], axis='columns', inplace=True)

val_df = data_test[:200000]
test_df = data_test[200000:]

x_train = data_train['review'].values
y_train = data_train['label'].values

x_val = val_df['review'].values
y_val = val_df['label'].values

x_test = test_df['review'].values
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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('Test score : ', score)
print('Test accuracy : ', acc)

# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()
#
# # using svg visual model
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# spent to time
print("--- %s seconds ---" % (time.time() - start_time))