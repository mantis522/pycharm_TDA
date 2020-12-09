import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import time

original_train_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\train_20000.csv", names=['id', 'label', 'review'])
original_test_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\test.csv", names=['id', 'label', 'review'])
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"
original_train_df = original_train_df.drop(original_train_df.index[0]).reset_index(drop=True)
original_test_df = original_test_df.drop(original_test_df.index[0]).reset_index(drop=True)
del original_test_df['id']
del original_train_df['id']

removed_amod_neg = r"D:\data\json_data\removed_data\popcorn\removed_amod_neg.json"
removed_amod_pos = r"D:\data\json_data\removed_data\popcorn\removed_amod_pos.json"
removed_PP_neg = r"D:\data\json_data\removed_data\popcorn\removed_PP_neg.json"
removed_PP_pos = r"D:\data\json_data\removed_data\popcorn\removed_PP_pos.json"
removed_SBAR_neg = r"D:\data\json_data\removed_data\popcorn\removed_SBAR_neg.json"
removed_SBAR_pos = r"D:\data\json_data\removed_data\popcorn\removed_SBAR_pos.json"

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
    df.columns = ['review']
    df['label'] = label

    return df

removed_amod_pos = making_df(removed_amod_pos, 1)
removed_amod_neg = making_df(removed_amod_neg, 0)
removed_PP_pos = making_df(removed_PP_pos, 1)
removed_PP_neg = making_df(removed_PP_neg, 0)
removed_SBAR_neg = making_df(removed_SBAR_neg, 0)
removed_SBAR_pos = making_df(removed_SBAR_pos, 1)

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)

concat_train_df = pd.concat([original_train_df]).reset_index(drop=True)

x_train = concat_train_df['review'].values
y_train = concat_train_df['label'].values
x_val = original_val_df['review'].values
y_val = original_val_df['label'].values
x_test = original_test_df['review'].values
y_test = original_test_df['label'].values

vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

maxlen = 80

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
y_train = to_categorical(np.asarray(y_train))
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
y_val = to_categorical(np.asarray(y_val))
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_test = to_categorical(np.asarray(y_test))

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print('Test score : ', score)
print('Test accuracy : ', acc)
