import json
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import re
import json
import random
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

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

tokenizer.fit_on_texts(x)