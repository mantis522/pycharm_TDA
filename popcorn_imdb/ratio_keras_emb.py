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

popcorn_parsed_neg = r"D:\data\json_data\parsed_data\popcorn_imdb\ratio\popcorn_parsed_10_neg.json"
popcorn_parsed_pos = r"D:\data\json_data\parsed_data\popcorn_imdb\ratio\popcorn_parsed_10_pos.json"
original_test_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\test.csv", names=['id', 'label', 'review'])
original_test_df = original_test_df.drop(original_test_df.index[0]).reset_index(drop=True)
del original_test_df['id']

def making_df(file_dir, label):
    with open(file_dir) as json_file:
        json_data = json.load(json_file)
        splited_sentence = json_data['splited_sentence']
        parsed_sentence = json_data['parsed_sentence']

    splited_sentence = splited_sentence[0]
    parsed_sentence = parsed_sentence[0]
    sent_list = []

    for a in range(len(splited_sentence)):
        sentence = " ".join(splited_sentence[a])
        sent_list.append(sentence)

    array1 = np.array(sent_list)

    col_name = ['review']
    df = pd.DataFrame(array1, columns=col_name)
    df['label'] = label

    return df

def making_test_df(file_directory, label):
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

ratio_pos = making_df(popcorn_parsed_pos, 1)
ratio_neg = making_df(popcorn_parsed_neg, 0)

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)
concat_train_df = pd.concat([ratio_neg, ratio_pos]).reset_index(drop=True)

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
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64, verbose=1, callbacks=[es, mc])
score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print('Test score : ', score)
print('Test accuracy : ', acc)