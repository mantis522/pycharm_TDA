import pandas as pd

data2 = [5, 6, 7, 8]
data3 = [9, 10, 11, 12]

df = pd.DataFrame({'review':data2})
df['new_col'] = data3

df = df.sample(frac=1).reset_index(drop=True)

print(df)