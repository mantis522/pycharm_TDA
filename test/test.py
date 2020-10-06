import json
import pandas as pd
import time


start_time = time.time()

origin_dir = "D:/data/json_data/train_data_full.json"
test_dir = "D:/data/json_data/test_data_full.json"
pos_gen_dir = "../reading_json/positive_EX.txt"
pos_gen_sentence_dir = "../reading_json/positive_EX_sentence.txt"
neg_gen_dir = "C:/Users/ruin/PycharmProjects/text_generator/neg_generate.txt"
neg_gen_sentence_dir = "C:/Users/ruin/PycharmProjects/text_generator/neg_sentence.txt"

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

    df_aug = pd.DataFrame(aug_list, columns=['data'])
    df_aug['label'] = labels


    return df_aug

print(making_augmented_df(neg_gen_dir, 0))