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

train_dir = r"D:\data\word2vec-nlp-tutorial\train_20000.csv"
test_dir = r"D:\data\word2vec-nlp-tutorial\test.csv"

original_train_df = pd.read_csv(train_dir, names=['id', 'label', 'review'])
original_test_df = pd.read_csv(test_dir, names=['id', 'label', 'review'])
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"
original_train_df = original_train_df.drop(original_train_df.index[0]).reset_index(drop=True)
original_test_df = original_test_df.drop(original_test_df.index[0]).reset_index(drop=True)
del original_test_df['id']
del original_train_df['id']

SEQ_LEN = 320
BATCH_SIZE = 8
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

def load_train_data(path, dataframe):
    global tokenizer
    indices, labels = [], []
    df = pd.read_csv(path)
    df = df[['label', 'review']]
    # df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    # df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    # df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    # df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    df = pd.concat([df, dataframe]).reset_index(drop=True)
    for a in range(len(df)):
        text = df['review'][a]
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
    del df['id']
    for a in range(len(df)):
        text = df['txt'][a]
        label = df['sentiment'][a]
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

train_x, train_y = load_test_data(train_dir)
test_x, test_y = load_test_data(test_dir)

val_x = [test_x[0][2000:], test_x[1][2000:]]
test_x = [test_x[0][:2000], test_x[1][:2000]]

test_y, val_y = test_y[:2000], test_y[2000:]

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
    outputs = keras.layers.Dense(units=2, activation='softmax')(dense)
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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
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