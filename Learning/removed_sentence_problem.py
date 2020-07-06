import json
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
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

removed_neg_PP = making_df(RPPN_directory, 0)
removed_pos_PP = making_df(RRPP_directory, 1)

# removed_pos_SBAR = making_df(RSBARN_directory, 0)
# removed_neg_SBAR = making_df(RSBARP_directory, 1)
#
# removed_neg_JJ = making_df(RN_directory, 0)
# removed_pos_JJ = making_df(RP_directory, 1)

test_df = making_test_df(test_directory)

removed_train_df = pd.concat([removed_neg_PP, removed_pos_PP])
removed_train_df = removed_train_df.reset_index(drop=True)

concat_train_df = pd.concat([removed_train_df, origin_train_df])
concat_train_df = concat_train_df.reset_index(drop=True)

# print(concat_train_df)
X_train = concat_train_df['data'].values
y_train = concat_train_df['label'].values

X_val = test_df['data'].values
y_val = test_df['label'].values

vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

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
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
hist = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), nb_epoch=2, batch_size=64, verbose=1)
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
