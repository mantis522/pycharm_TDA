import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

origin_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_train.csv"
test_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_test.csv"

home_glove_dir = "D:/ruin/data/glove.6B/glove.6B.100d.txt"

pos_gen_dir = "D:/ruin/data/gen_txt/Twitter_US_Sentiment/tweet_pos_generate.txt"
pos_gen_sentence_dir = "D:/ruin/data/gen_txt/Twitter_US_Sentiment/tweet_pos_sentence.txt"
neg_gen_dir = "D:/ruin/data/gen_txt/Twitter_US_Sentiment/tweet_neg_generate.txt"
neg_gen_sentence_dir = "D:/ruin/data/gen_txt/Twitter_US_Sentiment/tweet_neg_sentence.txt"
neu_gen_dir = "D:/ruin/data/gen_txt/Twitter_US_Sentiment/tweet_neu_generate.txt"
neu_gen_sentence_dir = "D:/ruin/data/gen_txt/Twitter_US_Sentiment/tweet_neu_sentence.txt"

def making_augmented_df(file_dir, labels):
    aug_list = []
    f = open(file_dir, 'r', encoding='utf-8')
    data = f.read()
    data = data.rsplit('\n')
    for a in range(len(data)):
        if data[a] != '':
            aug_list.append(data[a])
        else:
            pass
    f.close()

    aug_list = set(aug_list)

    df_aug = pd.DataFrame(aug_list, columns=['text'])
    df_aug['label'] = labels
    df_aug = df_aug.sample(frac=1).reset_index(drop=True)

    return df_aug

# c = making_augmented_df('C:/Users/ruin/PycharmProjects/text_generator/neg_generate.txt', 0)
# print(c)

def original_df(file_directory):
    df = pd.read_csv(file_directory)
    df = df[['airline_sentiment', 'text']]
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    # df.rename(columns={"text": "data"}, inplace=True)

    return df

pos_generation = making_augmented_df(pos_gen_sentence_dir, 2)
neg_generation = making_augmented_df(neg_gen_sentence_dir, 0)
neu_generation = making_augmented_df(neu_gen_sentence_dir, 1)

original_train_df = original_df(origin_dir)
original_test_df = original_df(test_dir)

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)

x = original_train_df['text']
y = original_train_df['label']

t = Tokenizer()
t.fit_on_texts(x)

vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(x)

def max_tweet():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

tweet_num = max_tweet()
maxlen = tweet_num

padded_X = pad_sequences(sequences, padding='post', maxlen=maxlen)

origin_train_df = original_df(origin_dir)

augmented_train_df = pd.concat(([pos_generation, neg_generation, neu_generation])).reset_index(drop=True)
concat_train_df = pd.concat(([augmented_train_df, origin_train_df])).reset_index(drop=True)

print(concat_train_df)

x_train = concat_train_df['text'].values
x_train = t.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
y_train = concat_train_df['label'].values
y_train = to_categorical(np.asarray(y_train))

x_test = original_test_df['text'].values
x_test = t.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_test = original_test_df['label'].values
y_test = to_categorical(np.asarray(y_test))

x_val = original_val_df['text'].values
x_val = t.texts_to_sequences(x_val)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
y_val = original_val_df['label'].values
y_val = to_categorical(np.asarray(y_val))

print('X_train size:', x_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', x_test.shape)
print('y_test size:', y_test.shape)
print('X_val size: ', x_val.shape)
print('y_val size: ', y_val.shape)

embeddings_index = {}
f = open(home_glove_dir, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100))

# fill in matrix
for word, i in t.word_index.items():  # dictionary
    embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe
    if embedding_vector is not None:
        # add to matrix
        embedding_matrix[i] = embedding_vector # each row of matrix

embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                           input_length = tweet_num, trainable=False)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist_1 = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, verbose=1, callbacks=[es, mc])

print("Accuracy...")
loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))