import os
import contextlib
import tensorflow as tf
import os
import codecs
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python import keras
import tensorflow.keras.backend as K
from keras_bert import Tokenizer
from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint
from sklearn.metrics import accuracy_score, f1_score

USE_TPU = False

if USE_TPU:
  TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
  tf.contrib.distribute.initialize_tpu_system(resolver)
  strategy = tf.contrib.distribute.TPUStrategy(resolver)

dataset = tf.keras.utils.get_file(
    fname="20news-18828.tar.gz",
    origin="http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz",
    extract=True,
)


SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
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


def load_data(path, tagset):
    """
    Input:
      path: Root directory where the categorical data sit in folders.
      tagset: List of folder-name, category tuples.
    Output:
      train_x / test_x: List with two items (viz token_input, seg_input)
      test_x / test_y: Output labels corresponding to trainx / train_y.
    """
    global tokenizer
    indices, labels = [], []
    for folder, label in tagset:
        folder = os.path.join(path, folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name), 'r', encoding="utf-8", errors='ignore') as reader:
                text = reader.read()
            ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)
            labels.append(label)

    items = list(zip(indices, labels))
    np.random.shuffle(items)
    indices, labels = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % BATCH_SIZE
    if mod > 0:
        indices, labels = indices[:-mod], labels[:-mod]
    return [indices, np.zeros_like(indices)], np.array(labels)


path = os.path.join(os.path.dirname(dataset), '20news-18828')
tagset = [(x, i) for i, x in enumerate(os.listdir(path))]
id_to_labels = {id_: label for label, id_ in tagset}

all_x, all_y = load_data(path, tagset)

train_perc = 0.8
total = len(all_y)

n_train = int(train_perc * total)
n_test = (total - n_train)

test_x = [all_x[0][n_train:], all_x[1][n_train:]]
train_x = [all_x[0][:n_train], all_x[1][:n_train]]

train_y, test_y = all_y[:n_train], all_y[n_train:]

print("# Total: %s, # Train: %s, # Test: %s" % (total, n_train, n_test))

# Load pretrained model
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
    outputs = keras.layers.Dense(units=20, activation='softmax')(dense)
    model = keras.models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

print(model.summary())

# sess = K.get_session()
# uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
# init_op = tf.variables_initializer(
#     [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
# )
# sess.run(init_op)

history = model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    shuffle=True,
    verbose=1
)

predicts = model.predict(test_x, verbose=True).argmax(axis=-1)
accuracy = accuracy_score(test_y, predicts)
macro_f1 = f1_score(test_y, predicts, average='macro')
micro_f1 = f1_score(test_y, predicts, average='micro')
weighted_f1 = f1_score(test_y, predicts, average='weighted')

print("Accuracy: %s" % accuracy)
print ('macro_f1: %s\nmicro_f1:%s\nweighted_f1:%s' %(
    macro_f1, micro_f1, weighted_f1)
)