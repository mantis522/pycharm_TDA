import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import re
import json
import random
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras import preprocessing
from keras.preprocessing.text import Tokenizer

origin_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_train.csv"
test_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_test.csv"

# origin_dir = "D:/data/json_data/train_data_full.json"
# test_dir = "D:/data/json_data/test_data_full.json"

# home_RPPN_directory = "D:/ruin/data/json_data/removed_data/removed_PP_neg.json"
# home_RRPP_directory = "D:/ruin/data/json_data/removed_data/removed_PP_pos.json"


def original_df(file_directory):
    df = pd.read_csv(file_directory, encoding='utf-8')
    df = df[['airline_sentiment', 'text']]
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    # df.rename(columns={"text": "data"}, inplace=True)

    return df

origin_train_df = original_df(origin_dir)
test_df = original_df(test_dir)
test_df = test_df.sample(frac=1).reset_index(drop=True)
# removed_neg_PP = making_df(home_RPPN_directory, 0)
# removed_pos_PP = making_df(home_RRPP_directory, 1)

# removed_train_df = pd.concat([removed_neg_PP, removed_pos_PP]).reset_index(drop=True)
# concat_train_df = pd.concat([removed_train_df, origin_train_df]).reset_index(drop=True)

TAG_RE = re.compile(r'<[^>]+>')


def clean_text(sentence):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    sentence = TAG_RE.sub('', sentence)

    return sentence

origin_train_df['clean_reviews'] = origin_train_df['text'].astype(str).apply(clean_text)
test_df['clean_reviews'] = test_df['text'].astype(str).apply(clean_text)

print(origin_train_df)
print(test_df)

y_train = origin_train_df['label']
y_test = test_df['label']

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

tokenized_reviews = [tokenize_reviews(data) for data in origin_train_df.text]
reviews_with_len = [[review, y_train[i], len(review)]
                 for i, review in enumerate(tokenized_reviews)]

tokenized_reviews2 = [tokenize_reviews(data) for data in test_df.text]
reviews_with_len2 = [[review, y_test[i], len(review)]
                 for i, review in enumerate(tokenized_reviews2)]


random.shuffle(reviews_with_len)

reviews_with_len.sort(key=lambda x: x[2])
reviews_with_len2.sort(key=lambda x: x[2])
sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]
sorted_reviews_labels2 = [(review_lab2[0], review_lab2[1]) for review_lab2 in reviews_with_len2]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))
processed_dataset2 = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels2, output_types=(tf.int32, tf.int32))
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
batched_dataset2 = processed_dataset2.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

train_data = batched_dataset
test_data = batched_dataset2


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=3,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 3

DROPOUT_RATE = 0.2

NB_EPOCHS = 10

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="categorical_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])


text_model.fit(train_data, epochs=NB_EPOCHS, batch_size=64, verbose=1)


score, acc = text_model.evaluate(test_data)
print('Test score : ', score)
print('Test acc : ', acc)