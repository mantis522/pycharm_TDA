import pandas as pd
from nltk.tokenize import word_tokenize

data_train = pd.read_csv("D:/data/amazon_review_polarity_csv/train.csv", names=['label', 'title', 'review'])
data_train.loc[data_train["label"] == 1, "label"] = 0
data_train.loc[data_train['label'] == 2, "label"] = 1
data_train.drop(['title'], axis='columns', inplace=True)
data_train = data_train.sample(frac=1).reset_index(drop=True)

data_test = pd.read_csv("D:/data/amazon_review_polarity_csv/test.csv", names=['label', 'title', 'review'])
data_test.loc[data_test["label"] == 1, "label"] = 0
data_test.loc[data_test["label"] == 2, "label"] = 1
data_test.drop(['title'], axis='columns', inplace=True)

def making_df(label, word_len, max_count):
    len_count = 0
    count = 0
    review_list = []
    label_list = []
    for a in range(len(data_train)):
        data_len = len(word_tokenize(data_train['review'][a]))
        if data_len > word_len and data_train['label'][a] == label:
            review_list.append(data_train['review'][a])
            label_list.append(data_train['label'][a])
            len_count = len_count + 1
            if len_count % 250 == 0:
                print("리스트수 : " + str(len_count))
            if len_count == max_count:
                break
        count = count + 1
        if count % 1000 == 0:
            print("전체 카운트수 : " + str(count))

    df = pd.DataFrame({'review': review_list})
    df['label'] = label_list
    print("카운트 끝")
    return df

len_rs = len(data_train) * 0.01

neg_df = making_df(0, 195, len_rs)
pos_df = making_df(1, 195, len_rs)

concat_train_df = pd.concat([neg_df, pos_df]).reset_index(drop=True)

concat_train_df.to_csv(r"D:\data\csv_file\amazon_len_per\one_percent.csv", index=False)