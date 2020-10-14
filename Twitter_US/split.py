import pandas as pd
from sklearn.model_selection import train_test_split
from pandas import DataFrame

df = pd.read_csv("D:/data/17_742210_compressed_Tweets.csv/Tweets.csv")

train, test = train_test_split(df, test_size=0.2, random_state=0)
train = DataFrame(train)
train.to_csv("D:/data/csv_file/airline_tweet_train.csv", encoding='utf-8')
test = DataFrame(test)
test.to_csv("D:/data/csv_file/airline_tweet_test.csv", encoding='utf-8')