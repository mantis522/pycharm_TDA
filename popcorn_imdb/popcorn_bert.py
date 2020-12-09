from transformers import BertTokenizer, TFBertModel
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

## something strange. check this later.

original_train_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\train_20000.csv", names=['id', 'label', 'review'])
original_test_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\test.csv", names=['id', 'label', 'review'])
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"
original_train_df = original_train_df.drop(original_train_df.index[0]).reset_index(drop=True)
original_test_df = original_test_df.drop(original_test_df.index[0]).reset_index(drop=True)
del original_test_df['id']
del original_train_df['id']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

x = original_train_df['review']
y = original_train_df['label']

t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(x)

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)

def bert_tokenizer(sent, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
        text = sent,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True
    )

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id

def max_text():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

text_num = max_text()
MAX_LEN = 320

def preprocessing(data_label, data_review):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    train_data_labels = []
    for train_label, train_sent in zip(data_label, data_review):
        try:
            input_id, attention_mask, token_type_id = bert_tokenizer(train_sent, MAX_LEN)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            train_data_labels.append(train_label)
        except Exception as e:
            print(e)
            print(train_sent)
            pass

    return input_ids, attention_masks, token_type_ids, train_data_labels

train_movie_input_ids = np.asarray(preprocessing(original_train_df['label'], original_train_df['review'])[0], dtype=int)
train_movie_attention_masks = np.asarray(preprocessing(original_train_df['label'], original_train_df['review'])[1], dtype=int)
train_movie_type_ids = np.asarray(preprocessing(original_train_df['label'], original_train_df['review'])[2], dtype=int)
train_movie_inputs = (train_movie_input_ids, train_movie_attention_masks, train_movie_type_ids)
train_data_labels = np.asarray(preprocessing(original_train_df['label'], original_train_df['review'])[3], dtype=int)

test_movie_input_ids = np.asarray(preprocessing(original_test_df['label'], original_test_df['review'])[0], dtype=int)
test_movie_attention_masks = np.asarray(preprocessing(original_test_df['label'], original_test_df['review'])[1], dtype=int)
test_movie_type_ids = np.asarray(preprocessing(original_test_df['label'], original_test_df['review'])[2], dtype=int)
test_movie_inputs = (test_movie_input_ids, test_movie_attention_masks, test_movie_type_ids)
test_data_labels = np.asarray(preprocessing(original_test_df['label'], original_test_df['review'])[3], dtype=int)

val_movie_input_ids = np.asarray(preprocessing(original_val_df['label'], original_val_df['review'])[0], dtype=int)
val_movie_attention_masks = np.asarray(preprocessing(original_val_df['label'], original_val_df['review'])[1], dtype=int)
val_movie_type_ids = np.asarray(preprocessing(original_val_df['label'], original_val_df['review'])[2], dtype=int)
val_movie_inputs = (val_movie_input_ids, val_movie_attention_masks, val_movie_type_ids)
val_data_labels = np.asarray(preprocessing(original_val_df['label'], original_val_df['review'])[3], dtype=int)

print("# sent : {}, # labels : {}".format(len(train_movie_input_ids), len(train_data_labels)))
print("# sent : {}, # labels : {}".format(len(test_movie_input_ids), len(test_data_labels)))
print("# sent : {}, # labels : {}".format(len(val_movie_input_ids), len(val_data_labels)))

class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()

        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), name="classifier")

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits

cls_model = TFBertClassifier(model_name="bert-base-uncased", dir_path='../test/bert_ckpt', num_class=2)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)

DATA_IN_PATH = 'data_in/popcorn_imdb'
DATA_OUT_PATH = "../test/data_out/popcorn_imdb"
model_name = "tf2_imdb_test"
BATCH_SIZE = 8
NUM_EPOCHS = 30

checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

history = cls_model.fit(train_movie_inputs, train_data_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_movie_inputs, val_data_labels), callbacks=[earlystop_callback, cp_callback], verbose=1)

print("Accuracy...")
loss, accuracy = cls_model.evaluate(test_movie_inputs, test_data_labels, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))