import pandas as pd
import json
import re


test = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_test.csv"
test_lab = "D:/data/17_742210_compressed_Tweets.csv/Tweets.csv"

df = pd.read_csv(test_lab)
df_sample = df[['airline_sentiment', 'text']]
# df_sample.rename(columns={"airline_sentiment":"변경후"}, inplace=True)

print(df_sample['airline_sentiment'])

count = 0
for a in range(len(df_sample)):
    if df_sample['airline_sentiment'].values() == 'neutral':
        count = count + 1
print(count)