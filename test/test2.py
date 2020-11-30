import pandas as pd
from nltk.tokenize import word_tokenize

data_train = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\train.csv", names=['id', 'sentiment', 'txt'])

txt_data = data_train['txt']

sum = 0

for a in range(len(txt_data)):
    text = txt_data[a]
    text_len = len(word_tokenize(text))
    sum = sum + text_len

