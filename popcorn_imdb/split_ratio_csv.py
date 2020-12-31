import pandas as pd
from tqdm import tqdm
import re

original_train_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\train_20000.csv", names=['id', 'label', 'review'])
original_test_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\test.csv", names=['id', 'label', 'review'])
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"
original_train_df = original_train_df.drop(original_train_df.index[0]).reset_index(drop=True)
original_test_df = original_test_df.drop(original_test_df.index[0]).reset_index(drop=True)
del original_test_df['id']
del original_train_df['id']

sent_list = []
sent_label = []

for a in range(len(original_train_df)):
    if original_train_df['label'][a] == '1':
        sent_list.append(original_train_df['review'][a])
        sent_label.append(original_train_df['label'][a])

df = pd.DataFrame([x for x in zip(sent_list, sent_label)])
df.rename(columns={0: 'review', 1: 'label'}, inplace=True)

def making_list(ratio_of2):
    len_of_sen = len(df)
    ra = int(len_of_sen) / 100 * ratio_of2
    ra = int(ra)

    return ra

ratio = making_list(25)

sliced_df = df[:ratio]

sliced_df.to_csv(r"D:\data\csv_file\popcorn\popcorn_25_pos.csv")