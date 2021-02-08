import json
import pandas as pd

file_in = r"D:\data\train_data_full.json"

with open(file_in) as json_file:
    json_data = json.load(json_file)

    review_data = json_data['data']

review_txt = []
review_label = []

for a in range(len(review_data)):
    review_label.append(review_data[a]['label'])
    review_txt.append(review_data[a]['txt'])

df = pd.DataFrame([x for x in zip(review_txt, review_label)])
df.rename(columns={0: 'review', 1: 'label'}, inplace=True)
# print(df)

df_pos = df[:12500]
df_neg = df[12500:]

ratio_pos_df = df_pos[:6250]
ratio_neg_df = df_neg[:6250]

concat_df = pd.concat([ratio_neg_df, ratio_pos_df]).reset_index(drop=True)

concat_df.to_csv(r"D:\data\csv_file\imdb\imdb_50.csv")