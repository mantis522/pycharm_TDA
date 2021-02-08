import json
import pandas as pd

original_train_neg_df = pd.read_csv(r"D:\data\csv_file\popcorn\popcorn_50_neg.csv")
original_train_pos_df = pd.read_csv(r"D:\data\csv_file\popcorn\popcorn_50_pos.csv")
original_concat_df = pd.concat([original_train_neg_df, original_train_pos_df]).reset_index(drop=True)
original_test_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\test.csv", names=['id', 'label', 'review'])
original_test_df = original_test_df.drop(original_test_df.index[0]).reset_index(drop=True)
del original_test_df['id']
del original_concat_df['Unnamed: 0']

removed_amod_neg = r"D:\data\json_data\removed_data\popcorn\ratio\50\removed_popcorn_50_amod_neg.json"
removed_amod_pos = r"D:\data\json_data\removed_data\popcorn\ratio\50\removed_popcorn_50_amod_pos.json"
removed_PP_neg = r"D:\data\json_data\removed_data\popcorn\ratio\50\removed_popcorn_50_PP_neg.json"
removed_PP_pos = r"D:\data\json_data\removed_data\popcorn\ratio\50\removed_popcorn_50_PP_pos.json"
removed_SBAR_neg = r"D:\data\json_data\removed_data\popcorn\ratio\50\removed_popcorn_50_SBAR_neg.json"
removed_SBAR_pos = r"D:\data\json_data\removed_data\popcorn\ratio\50\removed_popcorn_50_SBAR_pos.json"

def making_df(file_directory, label):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    removed_sentence = json_data['removed_sentence']
    removed_sentence = removed_sentence[0]
    removed = []

    for a in range(len(removed_sentence)):
        for b in range(len(removed_sentence[a])):
            removed.append(removed_sentence[a][b])

    df = pd.DataFrame(removed)
    df.columns = ['review']
    df['label'] = label

    return df

def making_test_df(file_directory, label):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    removed_sentence = json_data['removed_sentence']
    removed_sentence = removed_sentence[0]
    removed = []

    for a in range(len(removed_sentence)):
        for b in range(len(removed_sentence[a])):
            removed.append(removed_sentence[a][b])

    df = pd.DataFrame(removed)
    df.columns = ['review']
    df['label'] = label

    return df

removed_amod_pos = making_df(removed_amod_pos, 1)
removed_amod_neg = making_df(removed_amod_neg, 0)
removed_PP_pos = making_df(removed_PP_pos, 1)
removed_PP_neg = making_df(removed_PP_neg, 0)
removed_SBAR_neg = making_df(removed_SBAR_neg, 0)
removed_SBAR_pos = making_df(removed_SBAR_pos, 1)

concat_train_df = pd.concat([original_concat_df, removed_SBAR_neg, removed_SBAR_pos, removed_PP_pos, removed_PP_neg]).reset_index(drop=True)

print(len(concat_train_df))