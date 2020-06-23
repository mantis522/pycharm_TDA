#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json

with open("C:/Users/ruin/Desktop/IPython/deps/Data_Augmentation/test_data_full.json") as json_file2:
    json_data2 = json.load(json_file2)
    
test_data2 = json_data2['data']


# In[3]:


test_review2 = []
test_label2 = []

for a in range(len(test_data2)):
    test_review2.append(test_data2[a]['txt'])
    test_label2.append(test_data2[a]['label'])

import pandas as pd

df_ = pd.DataFrame(test_review2, columns=['data'])
df_['label'] = test_label2

with open("C:/Users/ruin/IdeaProjects/core-nlp-example/src/main/resources/train_neg.json") as json_file:
    json_data = json.load(json_file)

json_string = json_data["data"]

neg_list = []

for b in range(len(json_string)):
    item = json_string[b]['txt']
    neg_list.append(item)
    
df = pd.DataFrame(neg_list)
df.columns = ['data']
df['label'] = 0

with open("C:/Users/ruin/IdeaProjects/core-nlp-example/src/main/resources/train_pos.json") as json_file3:
    json_data3 = json.load(json_file3)
    print(json_data3)
    
pos_list = []
json_string3 = json_data3["data"]

for a in range(len(json_string3)):
    item = json_string3[a]['txt']
    pos_list.append(item)
    
df4 = pd.DataFrame(pos_list)
df4.columns = ['data']
df4['label'] = 1


# In[23]:


train_df = pd.concat([df, df4])


# In[24]:


train_df = train_df.reset_index(drop=True)


# In[25]:


train_df


# In[26]:


review_data = []
review_label = []

for i in range(len(train_df)):
    review_data.append(train_df['data'][i])
    review_label.append(train_df['label'][i])


# In[27]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 500
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(review_data)
X_train = tokenizer.texts_to_sequences(review_data)
X_train = sequence.pad_sequences(X_train, maxlen=max_features)
Y_train = review_label


# In[28]:


test_data = []
test_label = []

for k in range(len(df_)):
    test_data.append(df_['data'][k])
    test_label.append(df_['label'][k])

tokenizer.fit_on_texts(test_data)
X_test = tokenizer.texts_to_sequences(test_data)
X_test = sequence.pad_sequences(X_test, maxlen=max_features)
Y_test = test_label


# In[29]:


import numpy as np

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


# In[30]:


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


# In[31]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[32]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64, callbacks=[es, mc])


# In[ ]:




