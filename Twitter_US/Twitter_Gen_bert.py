#@title Set environment variables
import os
import contextlib
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
USE_TPU = False
os.environ['TF_KERAS'] = '1'

# @title Initialize TPU Strategy
if USE_TPU:
  TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
  tf.contrib.distribute.initialize_tpu_system(resolver)
  strategy = tf.contrib.distribute.TPUStrategy(resolver)
import json
import codecs
import numpy as np
from tqdm import tqdm

# Tensorflow Imports
import tensorflow as tf
from tensorflow.python import keras
import tensorflow.keras.backend as K

# Keras-bert imports
from keras_radam import RAdam
from keras_bert import Tokenizer
from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint

os.environ['TF_KERAS'] = '1'
train_dir = "D:/data/csv_file/airline_tweet_train.csv"
test_dir = "D:/data/csv_file/airline_tweet_test.csv"

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

pos_gen_dir = "D:/data/get_txt/tweet_pos_generate.txt"
pos_gen_sentence_dir = "D:/data/get_txt/tweet_pos_sentence.txt"
neg_gen_dir = "D:/data/get_txt/tweet_neg_generate.txt"
neg_gen_sentence_dir = "D:/data/get_txt/tweet_neg_sentence.txt"
neu_gen_dir = "D:/data/get_txt/tweet_neu_generate.txt"
neu_gen_sentence_dir = "D:/data/get_txt/tweet_neu_sentence.txt"

pos_generation = making_augmented_df(pos_gen_sentence_dir, 2)
neg_generation = making_augmented_df(neg_gen_sentence_dir, 0)
neu_generation = making_augmented_df(neu_gen_sentence_dir, 1)

SEQ_LEN = 128
BATCH_SIZE = 64
EPOCHS = 20
LR = 2e-5

pretrained_path = "D:/data/uncased_L-12_H-768_A-12"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)

gen_train_df = pd.concat([pos_generation, neg_generation, neu_generation]).reset_index(drop=True)

def load_train_data(path, dataframe):
    global tokenizer
    indices, labels = [], []
    df = pd.read_csv(path)
    df = df[['airline_sentiment', 'text']]
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    df = pd.concat([df, dataframe]).reset_index(drop=True)
    for a in range(len(df)):
        text = df['text'][a]
        label = df['label'][a]
        ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
        indices.append(ids)
        labels.append(label)
    items = list(zip(indices, labels))
    np.random.shuffle(items)
    indices, labels = zip(*items)
    indices = np.array(indices)
    # mod = indices.shape[0] % BATCH_SIZE
    # if mod > 0:
    #     indices, labels = indices[:-mod], labels[:-mod]

    return [indices, np.zeros_like(indices)], np.array(labels)

def load_test_data(path):
    global tokenizer
    indices, labels = [], []
    df = pd.read_csv(path)
    df = df[['airline_sentiment', 'text']]
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    for a in range(len(df)):
        text = df['text'][a]
        label = df['label'][a]
        ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
        indices.append(ids)
        labels.append(label)
    items = list(zip(indices, labels))
    np.random.shuffle(items)
    indices, labels = zip(*items)
    indices = np.array(indices)
    # mod = indices.shape[0] % BATCH_SIZE
    # if mod > 0:
    #     indices, labels = indices[:-mod], labels[:-mod]

    return [indices, np.zeros_like(indices)], np.array(labels)

train_x, train_y = load_train_data(train_dir, gen_train_df)
test_x, test_y = load_test_data(test_dir)

val_x = [test_x[0][1756:], test_x[1][1756:]]
test_x = [test_x[0][:1756], test_x[1][:1756]]

test_y, val_y = test_y[:1756], test_y[1756:]

# print(len(test_x[0]))
# print(len(val_x[0]))

with strategy.scope() if USE_TPU else contextlib.suppress():
    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=True,
        trainable=True,
        seq_len=SEQ_LEN,
    )

    # Add dense layer for classification
    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs = keras.layers.Dense(units=3, activation='softmax')(dense)
    model = keras.models.Model(inputs, outputs)

    model.compile(
        RAdam(lr=LR),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

print(model.summary())

sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(val_x, val_y),
    shuffle=True,
    verbose=1,
    callbacks=[es, mc]
)

predicts = model.predict(test_x, verbose=True).argmax(axis=-1)

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(test_y, predicts)
macro_f1 = f1_score(test_y, predicts, average='macro')
micro_f1 = f1_score(test_y, predicts, average='micro')
weighted_f1 = f1_score(test_y, predicts, average='weighted')

print("Accuracy: %s" % accuracy)
print ('macro_f1: %s\nmicro_f1:%s\nweighted_f1:%s' %(
    macro_f1, micro_f1, weighted_f1)
)