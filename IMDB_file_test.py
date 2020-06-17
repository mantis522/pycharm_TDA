import json
import pandas as pd

train_review = []
train_label = []

with open("D:/ruin/data/json_data/full_data/train_data_full.json") as json_file:
    json_data = json.load(json_file)

    train_data = json_data['data']

    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])

    df_train = pd.DataFrame(train_review, columns=['data'])
    df_train['label'] = train_label

    json_file.close()


test_review = []
test_label = []

with open("D:/ruin/data/json_data/full_data/test_data_full.json") as json_file:
    json_data = json.load(json_file)

    test_data = json_data['data']

    for a in range(len(test_data)):
        test_review.append(test_data[a]['txt'])
        test_label.append(test_data[a]['label'])

    df_test = pd.DataFrame(test_review, columns=['data'])
    df_test['label'] = test_label

    json_file.close()

review_data = []
review_label = []

for i in range(len(df_train)):
    review_data.append(df_train['data'][i])
    review_label.append(df_train['label'][i])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 500
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(review_data)
X_train = tokenizer.texts_to_sequences(review_data)
X_train = sequence.pad_sequences(X_train, maxlen=max_features)
Y_train = review_label

test_data = []
test_label = []

for k in range(len(df_test)):
    test_data.append(df_test['data'][k])
    test_label.append(df_test['label'][k])

tokenizer.fit_on_texts(test_data)
X_test = tokenizer.texts_to_sequences(test_data)
X_test = sequence.pad_sequences(X_test, maxlen=max_features)
Y_test = test_label

import numpy as np

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


# In[ ]:


from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

model = Sequential()
model.add(Embedding(5000, 120))
model.add(LSTM(120))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64, callbacks=[es, mc])

