import json
import pandas as pd


train_dir = "D:/data/csv_file/airline_tweet_train.csv"
test_dir = "D:/data/csv_file/airline_tweet_test.csv"

removed_amod_neg = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_amod_neg.json"
removed_amod_neu = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_amod_neu.json"
removed_amod_pos = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_amod_pos.json"
removed_PP_neg = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_PP_neg.json"
removed_PP_neu = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_PP_neu.json"
removed_PP_pos = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_PP_pos.json"
removed_SBAR_neg = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_SBAR_neg.json"
removed_SBAR_neu = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_SBAR_neu.json"
removed_SBAR_pos = "D:/data/json_data/removed_data/Twitter_US_Airline/removed_SBAR_pos.json"

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
    df.columns = ['text']
    df['label'] = label

    return df

def original_df(file_directory):
    df = pd.read_csv(file_directory)
    df = df[['airline_sentiment', 'text']]
    df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    # df.rename(columns={"text": "data"}, inplace=True)

    return df

removed_amod_pos = making_df(removed_PP_pos, 2)
removed_amod_neu = making_df(removed_amod_neu, 1)
removed_amod_neg = making_df(removed_amod_neg, 0)

removed_PP_pos = making_df(removed_PP_pos, 2)
removed_PP_neu = making_df(removed_PP_neu, 1)
removed_PP_neg = making_df(removed_PP_neg, 0)

removed_SBAR_pos = making_df(removed_SBAR_pos, 2)
removed_SBAR_neu = making_df(removed_SBAR_neu, 1)
removed_SBAR_neg = making_df(removed_SBAR_neg, 0)

removed_train_df = pd.concat([removed_amod_neg, removed_amod_pos, removed_amod_neu, removed_SBAR_neg, removed_SBAR_pos, removed_SBAR_neu]).reset_index(drop=True)

print(removed_train_df)