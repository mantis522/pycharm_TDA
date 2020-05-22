from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from nltk.parse.stanford import StanfordParser
from nltk import Tree
from contractions import CONTRACTION_MAP
import numpy as np
from sklearn.utils import shuffle

# os.environ['CLASSPATH'] = "C:/Users/ruin/Desktop/data/stanford-parser-full-2018-10-17"
os.environ['CLASSPATH'] = '/Users/ruin/Desktop/data/stanford-parser-full-2018-10-17'
stanford_parser = StanfordParser(
    model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
)

# basic_dir = "C:/Users/ruin/Desktop/data/aclImdb_v1/aclImdb"
basic_dir = '/Users/ruin/Desktop/data/aclImdb_v1/aclImdb'

path_train_pos = os.path.join(basic_dir, 'train', 'pos')

data = {}

data['txt'] = []

for i in os.listdir(path_train_pos):
    path = os.path.join(path_train_pos, i)
    rst = open(path, "r", encoding="UTF-8").read()
    data['txt'].append(rst)


def cleanText(readData):
    # 텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('<br />', '', readData)
    text = re.sub('[-=+#/\:^$@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\…》]', '', text)

    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    ## contraction들 원래대로 만들어주는 함수.
    ## 구원타자시다.

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def make_dict(dir):
    data = {}
    data['txt'] = []

    for i in os.listdir(dir):
        path = os.path.join(dir, i)
        rst = open(path, "r", encoding="UTF-8").read()
        rst = expand_contractions(rst)
        rst = cleanText(rst)
        data['txt'].append(rst)

    return pd.DataFrame.from_dict(data)

train_pos = make_dict(os.path.join(basic_dir, 'train', 'pos'))
train_neg = make_dict(os.path.join(basic_dir, 'train', 'neg'))
test_pos = make_dict(os.path.join(basic_dir, 'test', 'pos'))
test_neg = make_dict(os.path.join(basic_dir, 'test', 'neg'))

train_pos['label'] = 1
train_neg['label'] = 0
test_pos['label'] = 1
test_neg['label'] = 0

train_data = pd.concat([train_pos, train_neg], ignore_index=True)
test_data = pd.concat([test_pos, test_neg], ignore_index=True)

train_neg[:1000].to_json('train_neg_1000.json', orient='table')
# train_neg[1001:2000].to_json('train_neg_2000.json', orient='table')
# train_neg[2001:3000].to_json('train_neg_3000.json', orient='table')
# train_neg[3001:4000].to_json('train_neg_4000.json', orient='table')
# train_neg[4001:5000].to_json('train_neg_5000.json', orient='table')
# train_neg[5001:6000].to_json('train_neg_6000.json', orient='table')
# train_neg[6001:7000].to_json('train_neg_7000.json', orient='table')
# train_neg[7001:8000].to_json('train_neg_8000.json', orient='table')
# train_neg[8001:9000].to_json('train_neg_9000.json', orient='table')
# train_neg[9001:10000].to_json('train_neg_10000.json', orient='table')
# train_neg[10001:11000].to_json('train_neg_11000.json', orient='table')
# train_neg[11001:12000].to_json('train_neg_12000.json', orient='table')
# train_neg[12000:].to_json('train_neg_12499.json', orient='table')