import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

original_train_df = pd.read_csv(r"D:\data\csv_file\imdb\imdb_25.csv")
del original_train_df['Unnamed: 0']
test_directory = r"D:\data\test_data_full.json"

removed_amod_neg = r"D:\data\json_data\removed_data\IMDB\25\imdb_25_amod_neg.json"
removed_amod_pos = r"D:\data\json_data\removed_data\IMDB\25\imdb_25_amod_pos.json"
removed_PP_neg = r"D:\data\json_data\removed_data\IMDB\25\imdb_25_PP_neg.json"
removed_PP_pos = r"D:\data\json_data\removed_data\IMDB\25\imdb_25_PP_pos.json"
removed_SBAR_neg = r"D:\data\json_data\removed_data\IMDB\25\imdb_25_SBAR_neg.json"
removed_SBAR_pos = r"D:\data\json_data\removed_data\IMDB\25\imdb_25_SBAR_pos.json"

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

def making_test_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    test_data = json_data['data']
    test_review = []
    test_label = []

    for a in range(len(test_data)):
        test_review.append(test_data[a]['txt'])
        test_label.append(test_data[a]['label'])

    df = pd.DataFrame(test_review, columns=['review'])
    df['label'] = test_label

    return df

test_df = making_test_df(test_directory)
test_df = test_df.sample(frac=1).reset_index(drop=True)
test_df, val_df = train_test_split(test_df, test_size=0.4, random_state=0)

concat_train_df = pd.concat([original_train_df]).reset_index(drop=True)

x_train = concat_train_df['review'].values
y_train = concat_train_df['label'].values
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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print('Test score : ', score)
print('Test accuracy : ', acc)