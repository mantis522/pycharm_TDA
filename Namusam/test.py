import stanfordnlp
import time
import re
from itertools import combinations
import json

## 1. 삭제해도 되는 형용사들 넣었다 뺐다 반복하면서 augmentation 할 것
## 2. 이것도 앞뒤로 문장 붙이면서 augmentation 진행할 것
## 3.



start = time.time()

nlp = stanfordnlp.Pipeline(processors='tokenize,pos,depparse')

with open("D:/ruin/data/json_data/data_augmentation_neo/neg/test3.json") as json_file:
    json_data = json.load(json_file)

    splited_sentence = json_data['splited_sentence']
    parsed_sentence = json_data['parsed_sentence']
    augmented_sentence = json_data['augmented_text']

for a in range(len(splited_sentence)):

    for b in range(len(splited_sentence[a])):
        word_list = []
        sentence = splited_sentence[a][b]
        doc = nlp(sentence) ## 모든 sentence에 대해 depparse 실행하면서 대상 찾음.

        for i, _ in enumerate(doc.sentences):
            for word in doc.sentences[i].words:
                if word.dependency_relation == 'amod':
                    word_list.append(word.text)

        if len(word_list) > 1:
            print(word_list) ## 삭제 대상 단어 리스트
            print(str(a + 1) + " 번째 리스트의 " +str(b + 1)  + " 번째 문장")
            for c in range(len(word_list)):
                number = list(combinations(word_list, c + 1))
                for word_tuple in range(len(number)):
                    t2l = list(number[word_tuple])
                    sentencewords = sentence.split()

                    resultwords = [word for word in sentencewords if word.lower() not in t2l]
                    result = ' '.join(resultwords)
                    print(result)


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간