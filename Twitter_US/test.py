import pandas as pd


origin_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_train.csv"
test_dir = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_test.csv"

def original_df(file_directory):
    df = pd.read_csv(file_directory)
    df = df[['airline_sentiment', 'text']]
    # df['airline_sentiment'] = df['airline_sentiment'].replace('negative', 0)
    # df['airline_sentiment'] = df['airline_sentiment'].replace('neutral', 1)
    # df['airline_sentiment'] = df['airline_sentiment'].replace('positive', 2)
    # df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    # df.rename(columns={"text": "data"}, inplace=True)

    return df

count = 0
sum = 0

df = original_df(origin_dir)
print(df.loc[6]['airline_sentiment'])

# # print(df)
for a in range(len(df)):
    count = 0
    if df.loc[a]['airline_sentiment'] == 'neutral':
        count = count + 1
        sum = sum + count

print(sum)