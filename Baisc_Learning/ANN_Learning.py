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

origin_train_df = pd.concat([origin_train_df] * 10, ignore_index=True)

x_train = origin_train_df['data'].values
y_train = origin_train_df['label'].values

x_val = test_df['data'].values
y_val = test_df['label'].values

vocab_size = 1000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_val)

# print(X_train.shape)
# print(y_train.shape)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)

print(x_train[0])
print(y_train[0])

x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_val = tokenizer.sequences_to_matrix(x_val, mode='binary')

print(x_train.shape)

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

print(y_train.shape)
print(y_val.shape)

# model = Sequential()
# model.add(Dense(16, activation='relu', input_dim=1000))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.summary()
# # TODO: Compile the model using a loss function and an optimizer.
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model = Sequential()
model.add(Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
          activation='relu', input_dim=vocab_size))
model.add(Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
          activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy', keras.metrics.Recall(name='recall')])

hist = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_val, y_val)
                    ,verbose=2)

# score, acc = model.evaluate(x_val, y_val, batch_size=64, verbose=0)
# print("Test score : ", score[1])
# print("Test Accuracy : ", acc[1])

## 5. Training the model
# Run the model here. Experiment with different batch_size, and number of epochs!
# TODO: Run the model. Feel free to experiment with different batch sizes and number of epochs.
# hist = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val), verbose=2)

## 6. Evaluating the model
# This will give you the accuracy of the model,
# as evaluated on the testing set. Can you get something over 85%?
# score, acc = model.evaluate(x_val, y_val, batch_size=64, verbose=0)
#
# print("Accuracy: ", score)
# print('Test accuracy : ', acc)

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
