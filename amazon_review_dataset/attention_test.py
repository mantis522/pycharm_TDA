import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import time

## 돌아가는거 확인. 그런데 GPU memory가 낮으면 중간에 에러뜸. 12GB 서버에서는 제대로 돌아감. 8GB에서는 오락가락 하는 듯

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

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

embeddings_index = {}
f = open(glove_100_dir, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():  # dictionary
    embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe
    if embedding_vector is not None:
        # add to matrix
        embedding_matrix[i] = embedding_vector # each row of matrix

sequence_input = Input(shape=(maxlen, ), dtype='int32')
embedding_sequences = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                        input_length = text_num, trainable=False)(sequence_input)
#
lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_sequences)
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional\
    (LSTM(128, dropout=0.2, return_sequences=True, return_state=True))(lstm)

state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

attention = BahdanauAttention(128) # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.2)(dense1)
output = Dense(2, activation="softmax")(dropout)
model = Model(inputs=sequence_input, outputs=output)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_val, y_val), verbose=1, callbacks=[es, mc])

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

print("--- %s seconds ---" % (time.time() - start_time))