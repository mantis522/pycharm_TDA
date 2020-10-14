import pandas as pd
import json
import re

test = "D:/ruin/data/Twitter_US_Airline/original/airline_tweet_test.csv"

df = pd.read_csv(test)
df_sample = df[['airline_sentiment', 'text']]
df_sample.rename(columns={"airline_sentiment":"변경후"}, inplace=True)

print(df_sample)