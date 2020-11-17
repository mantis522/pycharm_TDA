import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, Dense, Flatten, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
import time
import tensorflow as tf

## 아직 안 돌아감

start_time = time.time()
K.clear_session()

train_dir = r"D:\data\csv_file\amazon_len_renew\amazon_10000_renew.csv"
test_dir = r"D:\data\csv_file\amazon_len_renew\amazon_test.csv"
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"

original_train_df = pd.read_csv(train_dir)
original_test_df = pd.read_csv(test_dir)

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)

x = original_train_df['review']
y = original_train_df['label']

t = Tokenizer()
t.fit_on_texts(x)

vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(x)


def max_text():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

text_num = max_text()
maxlen = text_num

padded_X = pad_sequences(sequences, padding='post', maxlen=maxlen)

x_train = original_train_df['review'].values
x_train = t.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
y_train = original_train_df['label'].values
y_train = to_categorical(np.asarray(y_train))

x_test = original_test_df['review'].values
x_test = t.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_test = original_test_df['label'].values
y_test = to_categorical(np.asarray(y_test))

x_val = original_val_df['review'].values
x_val = t.texts_to_sequences(x_val)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
y_val = original_val_df['label'].values
y_val = to_categorical(np.asarray(y_val))

embeddings_index = {}
f = open(glove_100_dir, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

model_name = 'amazon_classifier_en'
batch_size = 64
num_epochs = 10
max_len = max_text()

kargs = {
    'model_name': model_name,
    'vocab_size': vocab_size,
    'embedding_dimension': 100,
    'dropout_rate':0.2,
    'lstm_dimension':128,
    'output_dimension':2,
}

class amazonClassifier(tf.keras.Model):
    def __init__(self, **kargs):
        super(amazonClassifier, self).__init__(name=kargs['model_name'])
        self.embedding = Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embedding_dimension'])
        self.lstm_layer = LSTM(kargs['lstm_dimension'], return_sequences=True)
        self.dropout = Dropout(kargs['dropout_rate'])
        self.fc1 = Dense(units=kargs['output_dimension'], activation=tf.keras.activations.softmax)

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.lstm_layer(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

model = amazonClassifier(**kargs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()