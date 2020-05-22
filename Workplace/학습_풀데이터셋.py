import json

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/imi_2.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_imi_2.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_imi_2.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['augmented_text']

import pandas as pd

df_imi_2 = pd.DataFrame(json_string, columns=['data'])
df_imi_2['label'] = 1

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/imi_1.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_imi_1.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_imi_1.json", encoding='utf-8') as json_file2:
    json_data2 = json.load(json_file2)
    json_string2 = json_data2['augmented_text']

df_imi_1 = pd.DataFrame(json_string2, columns=['data'])
df_imi_1['label'] = 1

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/3rd.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_3rd.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_3rd.json", encoding='utf-8') as json_file3:
    json_data3 = json.load(json_file3)
    json_string3 = json_data3['augmented_text']

df_3rd = pd.DataFrame(json_string3, columns=['data'])
df_3rd['label'] = 1

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/4th.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_4th.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_4th.json", encoding='utf-8') as json_file4:
    json_data4 = json.load(json_file4)
    json_string4 = json_data4['augmented_text']

df_4th = pd.DataFrame(json_string4, columns=['data'])
df_4th['label'] = 1

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/4th.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_5th.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_5th.json", encoding='utf-8') as json_file5:
    json_data5 = json.load(json_file5)
    json_string5 = json_data5['augmented_text']

df_5th = pd.DataFrame(json_string5, columns=['data'])
df_5th['label'] = 1

train_pos_df = pd.concat([df_3rd, df_4th, df_5th, df_imi_1, df_imi_2])

train_pos_df = train_pos_df.reset_index(drop=True)

train_pos_df

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/neg_imi_1.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_neg_imi1.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_neg_imi1.json", encoding='utf-8') as json_file_:
    json_data_ = json.load(json_file_)
    json_string_ = json_data_['augmented_text']

df_neg_imi1 = pd.DataFrame(json_string_, columns=['data'])
df_neg_imi1['label'] = 0

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/neg_imi_2.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_neg_imi2.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_neg_imi2.json", encoding='utf-8') as json_file_2:
    json_data_2 = json.load(json_file_2)
    json_string_2 = json_data_2['augmented_text']

df_neg_imi2 = pd.DataFrame(json_string_2, columns=['data'])
df_neg_imi2['label'] = 0

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/neg_3rd.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_neg_3rd.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_neg_3rd.json", encoding='utf-8') as json_file_3:
    json_data_3 = json.load(json_file_3)
    json_string_3 = json_data_3['augmented_text']

df_neg_3rd = pd.DataFrame(json_string_3, columns=['data'])
df_neg_3rd['label'] = 0

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/neg_4th.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_neg_4th.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_neg_4th.json", encoding='utf-8') as json_file_4:
    json_data_4 = json.load(json_file_4)
    json_string_4 = json_data_4['augmented_text']

df_neg_4th = pd.DataFrame(json_string_4, columns=['data'])
df_neg_4th['label'] = 0

text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/neg_5th.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("aug_neg_5th.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()

with open("aug_neg_5th.json", encoding='utf-8') as json_file_5:
    json_data_5 = json.load(json_file_5)
    json_string_5 = json_data_5['augmented_text']

df_neg_5th = pd.DataFrame(json_string_5, columns=['data'])
df_neg_5th['label'] = 0

df_neg_5th

train_neg_df = pd.concat([df_neg_3rd, df_neg_4th, df_neg_5th, df_neg_imi1, df_neg_imi2])

train_neg_df = train_neg_df.reset_index(drop=True)

train_neg_df

train_pos_df

origin_neg = []

with open("C:/Users/ruin/IdeaProjects/core-nlp-example/src/main/resources/train_neg.json") as json_file_origin_neg:
    json_data_origin_neg = json.load(json_file_origin_neg)
    json_string_origin_neg = json_data_origin_neg["data"]

for a in range(len(json_string_origin_neg)):
    item = json_string_origin_neg[a]['txt']
    origin_neg.append(item)

origin_neg_df = pd.DataFrame(origin_neg)
origin_neg_df.columns = ['data']
origin_neg_df['label'] = 0

origin_pos = []

with open("C:/Users/ruin/IdeaProjects/core-nlp-example/src/main/resources/train_pos.json") as json_file_origin_pos:
    json_data_origin_pos = json.load(json_file_origin_pos)
    json_string_origin_pos = json_data_origin_pos["data"]

for a in range(len(json_string_origin_pos)):
    item = json_string_origin_pos[a]['txt']
    origin_pos.append(item)

origin_pos_df = pd.DataFrame(origin_pos)
origin_pos_df.columns = ['data']
origin_pos_df['label'] = 1

origin_1000_dataset = pd.concat([origin_neg_df, origin_pos_df])
origin_1000_dataset = origin_1000_dataset.reset_index(drop=True)
origin_1000_dataset

augmented_df = pd.concat([train_neg_df, train_pos_df])
augmented_df = augmented_df.reset_index(drop=True)

augmented_df

origin_augmented = pd.concat([origin_1000_dataset, augmented_df])
origin_augmented = origin_augmented.reset_index(drop=True)

train_neg_df = pd.concat([origin_neg_df, train_neg_df])

train_pos_df = pd.concat([origin_pos_df, train_pos_df])

train_df = pd.concat([train_neg_df, train_pos_df])

train_df = train_df.reset_index(drop=True)

with open("C:/Users/ruin/Desktop/IPython/deps/Data_Augmentation/train_data_full.json") as json_file:
    json_data = json.load(json_file)
    train_data = json_data['data']

train_review = []
train_label = []

for k in range(len(train_data)):
    train_review.append(train_data[k]['txt'])
    train_label.append(train_data[k]['label'])

df_train_data = pd.DataFrame(train_review, columns=['data'])
df_train_data['label'] = train_label

df_train_data

df_train_data_augmented_data = pd.concat([augmented_df, df_train_data])
df_train_data_augmented_data = df_train_data_augmented_data.reset_index(drop=True)

df_train_data_augmented_data

### 전체 트레이닝 셋 이용한 ##

# review_data = []
# review_label = []

# for i in range(len(df_train_data_augmented_data)):
#     review_data.append(df_train_data_augmented_data['data'][i])
#     review_label.append(df_train_data_augmented_data['label'][i])

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence

# max_features = 500
# tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
# tokenizer.fit_on_texts(review_data)
# X_train = tokenizer.texts_to_sequences(review_data)
# X_train = sequence.pad_sequences(X_train, maxlen=max_features)
# Y_train = review_label

### 일반 전체 트레이닝 셋 이용한 ###

review_data = []
review_label = []

for i in range(len(df_train_data)):
    review_data.append(df_train_data['data'][i])
    review_label.append(df_train_data['label'][i])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 500
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(review_data)
X_train = tokenizer.texts_to_sequences(review_data)
X_train = sequence.pad_sequences(X_train, maxlen=max_features)
Y_train = review_label

with open("C:/Users/ruin/Desktop/IPython/deps/Data_Augmentation/test_data_full.json") as json_file2:
    json_data2 = json.load(json_file2)

test_data2 = json_data2['data']

test_review2 = []
test_label2 = []

for a in range(len(test_data2)):
    test_review2.append(test_data2[a]['txt'])
    test_label2.append(test_data2[a]['label'])

import pandas as pd

df_ = pd.DataFrame(test_review2, columns=['data'])
df_['label'] = test_label2

test_data = []
test_label = []

for k in range(len(df_)):
    test_data.append(df_['data'][k])
    test_label.append(df_['label'][k])

tokenizer.fit_on_texts(test_data)
X_test = tokenizer.texts_to_sequences(test_data)
X_test = sequence.pad_sequences(X_test, maxlen=max_features)
Y_test = test_label

import numpy as np

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

model = Sequential()
model.add(Embedding(5000, 120))
model.add(LSTM(120))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('../best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64, callbacks=[es, mc])


