#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


with open("aug_imi_2.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['augmented_text']


# In[3]:


import pandas as pd

df_imi_2 = pd.DataFrame(json_string, columns=['data'])
df_imi_2['label'] = 1


# In[4]:


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


# In[5]:


with open("aug_imi_1.json", encoding='utf-8') as json_file2:
    json_data2 = json.load(json_file2)
    json_string2 = json_data2['augmented_text']
    
df_imi_1 = pd.DataFrame(json_string2, columns=['data'])
df_imi_1['label'] = 1


# In[6]:


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


# In[7]:


with open("aug_3rd.json", encoding='utf-8') as json_file3:
    json_data3 = json.load(json_file3)
    json_string3 = json_data3['augmented_text']
    
df_3rd = pd.DataFrame(json_string3, columns=['data'])
df_3rd['label'] = 1


# In[8]:


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


# In[9]:


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


# In[10]:


train_pos_df = pd.concat([df_3rd, df_4th, df_5th, df_imi_1, df_imi_2])


# In[11]:


train_pos_df = train_pos_df.reset_index(drop=True)


# In[12]:


train_pos_df


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


df_neg_5th


# In[19]:


train_neg_df = pd.concat([df_neg_3rd, df_neg_4th, df_neg_5th, df_neg_imi1, df_neg_imi2])


# In[20]:


train_neg_df = train_neg_df.reset_index(drop=True)


# In[21]:


train_neg_df


# In[22]:


train_pos_df


# In[23]:


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


# In[24]:


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


# In[25]:


augmented_df = pd.concat([train_neg_df, train_pos_df])
augmented_df = augmented_df.reset_index(drop=True)


# In[32]:


augmented_df


# In[26]:


train_neg_df = pd.concat([origin_neg_df, train_neg_df])


# In[27]:


train_pos_df = pd.concat([origin_pos_df, train_pos_df])


# In[28]:


train_df = pd.concat([train_neg_df, train_pos_df])


# In[29]:


train_df = train_df.reset_index(drop=True)


# In[30]:


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


# In[31]:


df_train_data


# In[33]:


df_train_data_augmented_data = pd.concat([augmented_df, df_train_data])
df_train_data_augmented_data = df_train_data_augmented_data.reset_index(drop=True)


# In[34]:


df_train_data_augmented_data


# In[35]:


review_data = []
review_label = []

for i in range(len(df_train_data_augmented_data)):
    review_data.append(df_train_data_augmented_data['data'][i])
    review_label.append(df_train_data_augmented_data['label'][i])
    
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 500
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(review_data)
X_train = tokenizer.texts_to_sequences(review_data)
X_train = sequence.pad_sequences(X_train, maxlen=max_features)
Y_train = review_label


# In[36]:


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


# In[37]:


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


# In[ ]:




