#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd

text_list = []
text_data = {}

# with open("D:/ruin/data/json_data/data_augmentation_neo/neg/EX_neg.json") as json_file:
#     json_data = json.load(json_file)
#     json_string = json_data["augmented_text"]
#     json_string2 = json_data["augmented_text2"]
#     json_string3 = json_data["augmented_text3"]
#
#     for a in json_string:
#         text = ' '.join(a)
#         text_list.append(text)
#
#     for b in json_string2:
#         text = ' '.join(b)
#         text_list.append(text)
#
#     for c in json_string3:
#         text = ' '.join(c)
#         text_list.append(text)
#
#     json_file.close()
    
# text_data['augmented_text'] = text_list
#
# with open("test.json", 'w', encoding='utf-8') as make_file:
#     json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
#     make_file.close()
    
with open("test.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['augmented_text']
    
df = pd.DataFrame(json_string, columns=['data'])
df['label'] = 0    


# In[3]:


# text_list2 = []
# text_data2 = {}
#
# with open("D:/ruin/data/json_data/data_augmentation_neo/pos/EX_pos.json") as json_file:
#     json_data = json.load(json_file)
#     json_string = json_data["augmented_text"]
#     json_string2 = json_data["augmented_text2"]
#     json_string3 = json_data["augmented_text3"]
#
#     for a in json_string:
#         text = ' '.join(a)
#         text_list2.append(text)
#
#     for b in json_string2:
#         text = ' '.join(b)
#         text_list2.append(text)
#
#     for c in json_string3:
#         text = ' '.join(c)
#         text_list2.append(text)
#
#     json_file.close()
#
# text_data2['augmented_text'] = text_list2


# with open("test2.json", 'w', encoding='utf-8') as make_file:
#     json.dump(text_data2, make_file, ensure_ascii=False, indent='\t')
#     make_file.close()
    
with open("test2.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['augmented_text']
    
    df2 = pd.DataFrame(json_string, columns=['data'])
    df2['label'] = 1
    
    json_file.close()


# In[4]:


df2


# In[5]:


df_augmented = pd.concat([df, df2])


# In[9]:


train_review = []
train_label = []

with open("D:/ruin/data/json_data/full_data/train_data_full.json") as json_file:
    json_data = json.load(json_file)
    
    train_data = json_data['data']
    
    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])
        
    df_train = pd.DataFrame(train_review, columns=['data'])
    df_train['label'] = train_label
    
    json_file.close()


# In[11]:


df_augmented = pd.concat([df_augmented, df_train])


# In[13]:


df_augmented = df_augmented.reset_index(drop=True)


# In[15]:


test_review = []
test_label = []

with open("D:/ruin/data/json_data/full_data/test_data_full.json") as json_file:
    json_data = json.load(json_file)
    
    test_data = json_data['data']
    
    for a in range(len(test_data)):
        test_review.append(test_data[a]['txt'])
        test_label.append(test_data[a]['label'])
        
    df_test = pd.DataFrame(test_review, columns=['data'])
    df_test['label'] = test_label
    
    json_file.close()


# In[18]:


review_data = []
review_label = []

for i in range(len(df_augmented)):
    review_data.append(df_augmented['data'][i])
    review_label.append(df_augmented['label'][i])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 500
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
tokenizer.fit_on_texts(review_data)
X_train = tokenizer.texts_to_sequences(review_data)
X_train = sequence.pad_sequences(X_train, maxlen=max_features)
Y_train = review_label

test_data = []
test_label = []

for k in range(len(df_test)):
    test_data.append(df_test['data'][k])
    test_label.append(df_test['label'][k])

tokenizer.fit_on_texts(test_data)
X_test = tokenizer.texts_to_sequences(test_data)
X_test = sequence.pad_sequences(X_test, maxlen=max_features)
Y_test = test_label

import numpy as np

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


# In[ ]:


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
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64, callbacks=[es, mc])

