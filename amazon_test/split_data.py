import pandas as pd
import json

data_train = pd.read_csv("D:/data/amazon_review_polarity_csv/train.csv", names=['label', 'title', 'review'])
data_train.loc[data_train["label"] == 1, "label"] = 0
data_train.loc[data_train['label'] == 2, "label"] = 1
data_train.drop(['title'], axis='columns', inplace=True)

data_test = pd.read_csv("D:/data/amazon_review_polarity_csv/test.csv", names=['label', 'title', 'review'])
data_test.loc[data_test["label"] == 1, "label"] = 0
data_test.loc[data_test["label"] == 2, "label"] = 1
data_test.drop(['title'], axis='columns', inplace=True)

def split_df(label, number):
    review_list = []
    label_list = []
    for a in range(len(data_test)):
        if data_test['label'][a] == label:
            label_list.append(data_test['label'][a])
            review_list.append(data_test['review'][a])
        if len(review_list) == number:
            break
    df = pd.DataFrame()
    df['label'] = label_list
    df['review'] = review_list

    return df

neg_df = split_df(0, 12500)
pos_df = split_df(1, 12500)

concat_train_df = pd.concat([neg_df, pos_df]).reset_index(drop=True)
print(len(concat_train_df))

concat_train_df.to_csv("D:/data/csv_file/amazon/amazon_test.csv", index=False)