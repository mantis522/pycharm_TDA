import json
import pandas as pd

removed_amod_neg = r"D:\data\json_data\removed_data\amazon\100000\old\removed_amod_neg_100000.json"
removed_amod_pos = r"D:\data\json_data\removed_data\amazon\100000\old\removed_amod_pos_100000.json"
removed_PP_neg = r"D:\data\json_data\removed_data\amazon\100000\old\removed_PP_neg_100000.json"
removed_PP_pos = r"D:\data\json_data\removed_data\amazon\100000\old\removed_PP_pos_100000.json"
removed_SBAR_neg = r"D:\data\json_data\removed_data\amazon\100000\old\removed_SBAR_neg_100000.json"
removed_SBAR_pos = r"D:\data\json_data\removed_data\amazon\100000\old\removed_SBAR_pos_100000.json"

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

removed_amod_pos = making_df(removed_amod_pos, 1)
removed_amod_neg = making_df(removed_amod_neg, 0)
removed_PP_pos = making_df(removed_PP_pos, 1)
removed_PP_neg = making_df(removed_PP_neg, 0)
removed_SBAR_neg = making_df(removed_SBAR_neg, 0)
removed_SBAR_pos = making_df(removed_SBAR_pos, 1)


concat_train_df = pd.concat([removed_amod_neg, removed_amod_pos, removed_SBAR_neg, removed_SBAR_pos, removed_PP_neg, removed_PP_pos]).reset_index(drop=True)

print(len(concat_train_df))