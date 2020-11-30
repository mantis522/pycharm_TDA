import pandas as pd

original_train_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\train.csv", names=['id', 'sentiment', 'txt'])
original_test_df = pd.read_csv(r"D:\data\word2vec-nlp-tutorial\test.csv", names=['id', 'sentiment', 'txt'])

original_test_df = original_test_df.drop(original_test_df.index[0])
print(original_test_df)