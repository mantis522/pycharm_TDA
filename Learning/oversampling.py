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

origin_neg_directory = "C:/Users/ruin/Desktop/data/json_data/train_neg_full.json"
origin_pos_directory = "C:/Users/ruin/Desktop/data/json_data/train_pos_full.json"
origin_directory = "C:/Users/ruin/Desktop/data/train_data_full.json"
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

origin_train_df = making_origin_df(origin_directory)
test_df = making_test_df(test_directory)

origin_train_df = pd.concat([origin_train_df] * 1, ignore_index=True)

# review_data = origin_train_df['data'].tolist()
# review_label = origin_train_df['label'].tolist()
# test_data = test_df['data'].tolist()
# test_label = test_df['label'].tolist()

X_train = origin_train_df['data'].values
y_train = origin_train_df['label'].values

X_val = test_df['data'].values
y_val = test_df['label'].values

vocab_size = 10000
# max_features = 500

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

l = [len(i) for i in X_train_seq]
l = np.array(l)

print('minimum number of words: {}'.format(l.min()))
print('median number of words: {}'.format(np.median(l)))
print('average number of words: {}'.format(l.mean()))
print('maximum number of words: {}'.format(l.max()))

print(X_train[0])
print(X_train_seq[0])

def get_keras_model(lstm_units, neurons_dense, dropout_rate, embedding_size, max_text_len):
    inputs = Input(shape=(max_text_len,))
    x = Embedding(vocab_size, embedding_size)(inputs)
    x = LSTM(units=lstm_units)(x)
    x = Dense(neurons_dense, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def keras_cv_score(parameterization, weight=None):
    max_text_len = parameterization.get('max_text_len')

    keras.backend.clear_session()
    model = get_keras_model(parameterization.get('lstm_units'),
                            parameterization.get('neurons_dense'),
                            parameterization.get('dropout_rate'),
                            parameterization.get('embedding_size'),
                            max_text_len)

    learning_rate = parameterization.get('learning_rate')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    NUM_EPOCHS = parameterization.get('num_epochs')

    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC()])

    X_train_seq_padded = pad_sequences(X_train_seq, maxlen=max_text_len)
    X_val_seq_padded = pad_sequences(X_val_seq, maxlen=max_text_len)

    res = model.fit(x=X_train_seq_padded,
                    y=y_train,
                    batch_size=parameterization.get('batch_size'),
                    epochs=NUM_EPOCHS,
                    validation_data=(X_val_seq_padded, y_val))

    last_score = np.array(res.history['val_auc_1'][-1:])
    return last_score, 0

parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.0001, 0.5],
        "log_scale": True,
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": True,
    },
    {
        "name": "lstm_units",
        "type": "range",
        "bounds": [1, 10],
        "value_type": "int"
    },
    {
        "name": "neurons_dense",
        "type": "range",
        "bounds": [1, 300],
        "value_type": "int"
    },
    {
        "name": "num_epochs",
        "type": "range",
        "bounds": [1, 20],
        "value_type": "int"
    },
    {
        "name": "batch_size",
        "type": "range",
        "bounds": [8, 64],
        "value_type": "int"
    },
    {
        "name": "embedding_size",
        "type": "range",
        "bounds": [2, 500],
        "value_type": "int"
    },
    {
        "name": "max_text_len",
        "type": "range",
        "bounds": [10, 800],
        "value_type": "int"
    },
]

from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

ax_client = AxClient()

ax_client.create_experiment(
    name="keras_experiment",
    parameters=parameters,
    objective_name='keras_cv',
    minimize=False)


def evaluate(parameters):
    return {"keras_cv": keras_cv_score(parameters)}

for i in range(2):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


ax_client.get_trials_data_frame().sort_values('trial_index')

best_parameters, values = ax_client.get_best_parameters()

for k in best_parameters.items():
    print(k)

print()

# the best score achieved.
means, covariances = values
print(means)

keras.backend.clear_session()

max_text_len = best_parameters['max_text_len']
model = get_keras_model(best_parameters['lstm_units'],
                        best_parameters['neurons_dense'],
                        best_parameters['dropout_rate'],
                        best_parameters['embedding_size'],
                        max_text_len)

optimizers = keras.optimizers.Adam(learning_rate=best_parameters['learning_rate'])

model.compile(optimizers=optimizers,
              loss=keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

tokenizer0 = Tokenizer(num_words=vocab_size)
tokenizer0.fit_on_texts(origin_train_df['data'].values)
X_train_seq0 = tokenizer0.texts_to_sequences(origin_train_df['data'].values)
X_train_seq0_padded = pad_sequences(X_train_seq0, maxlen=max_text_len)
y_train0 = origin_train_df['label'].values

model.fit(x=X_train_seq0_padded, y=y_train0, batch_size=best_parameters['batch_size'], epochs=best_parameters['num_epochs'])

# # tokenizer = Tokenizer(num_words=max_features, split=' ')
# # tokenizer.fit_on_texts(review_data)
# X_train = tokenizer.texts_to_sequences(review_data)
# X_train = sequence.pad_sequences(X_train, maxlen=max_features)
# Y_train = review_label
#
#
# embed_dim = 128
# lstm_out = 196
#
#
# tokenizer.fit_on_texts(test_data)
# X_test = tokenizer.texts_to_sequences(test_data)
# X_test = sequence.pad_sequences(X_test, maxlen=max_features)
# Y_test = test_label
#
# X_train = np.asarray(X_train).astype('float32')
# Y_train = np.array(Y_train).astype('float32')
# X_test = np.array(X_test).astype('float32')
# Y_test = np.array(Y_test).astype('float32')
#
# # print(X_train.shape, Y_train.shape)
# # print(X_test.shape, Y_train.shape)
#
# # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#
# model = Sequential()
# model.add(Embedding(max_features, embed_dim, input_length= X_train.shape[1]))
# model.add(SpatialDropout1D(0.4))
# model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, Y_train, epochs=7, batch_size=32, verbose=2)


# model = Sequential()
# model.add(Embedding(5000, 120))
# model.add(LSTM(120))
# model.add(Dense(1, activation='sigmoid'))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64, callbacks=[es, mc])


## https://www.justintodata.com/sentiment-analysis-with-deep-learning-lstm-keras-python/