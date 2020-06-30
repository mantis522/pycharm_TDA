import stanfordnlp
import time
import re
from itertools import combinations
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0

## 1. 삭제해도 되는 형용사들 넣었다 뺐다 반복하면서 augmentation 할 것
## 2. 이것도 앞뒤로 문장 붙이면서 augmentation 진행할 것
## 3. 감성분석해서 삭제한 문장이 원래 극성을 해치지 않으면

nlp = stanfordnlp.Pipeline(processors='tokenize,pos,depparse')

start = time.time()

TPE_list = []

with open("C:/Users/ruin/Desktop/data/json_data/TPE_Pattern/EX_pos.json") as json_file:
# with open("C:/Users/ruin/Desktop/data/json_data/TPE_Pattern/EX_pos.json") as json_file:
    json_data = json.load(json_file)

    splited_sentence = json_data['splited_sentence']
    parsed_sentence = json_data['parsed_sentence']
    augmented_sentence = json_data['augmented_text']

index_num = []

for a in range(len(augmented_sentence)):
    # print(augmented_sentence[a])
    for b in range(len(augmented_sentence[a])):
        if augmented_sentence[a][b] != None:
            index_num.append(a)
            # print(augmented_sentence[a])
            # print(a, b)

index_num = list(set(index_num))
sent_list = []

for a in index_num:
    data = splited_sentence[a]

    for b in range(len(data)):
        word_list = []

        list1 = data[:b]
        list2 = data[b + 1:]
        sentence = splited_sentence[a][b]

        doc = nlp(sentence)  ## 모든 sentence에 대해 remove 실행하면서 대상 찾음.

        for i, _ in enumerate(doc.sentences):
            for word in doc.sentences[i].words:
                if word.dependency_relation == 'amod':
                    word_list.append(word.text)

        if len(word_list) > 1:
            first_sent_list = []
            # print("기본 문장 : " + splited_sentence[a][b])
            print(str(a + 1) + " 번째 리스트의 " + str(b + 1) + " 번째 문장")
            # print("원래 문장에 대한 감성분석 결과값 : " + str(vader_polarity(sentence)))

            score_original = vader_polarity(sentence)  ## 원래 문장에 대한 감성분석 스코어

            print(word_list)  ## 삭제 대상 단어 리스트

            for c in range(len(word_list)):
                number = list(combinations(word_list, c + 1))
                for word_tuple in range(len(number)):
                    t2l = list(number[word_tuple])
                    sentencewords = sentence.split()
                    resultwords = [word for word in sentencewords if word.lower() not in t2l]
                    result = ' '.join(resultwords)
                    score_remove = vader_polarity(result)

                    if (score_original == score_remove):
                        result_sentence = ' '.join(list1) + " " + result + " " + ' '.join(list2)
                        # print("해당 문장에 대해서만 감성분석 결과값 : " + str(vader_polarity(result)))
                        print(result_sentence)
                        first_sent_list.append(result_sentence)

                    # print("전체 세그먼트에 대한 감성분석 결과값 : " + str(vader_polarity(result_sentence)))

            sent_list.append(first_sent_list)
    # print('------------------------')

print(sent_list)

sent_json = {}
sent_json['removed_sentence'] = []
sent_json['removed_sentence'].append(sent_list)

with open("C:/Users/ruin/Desktop/data/json_data/removed_data/test_pos.json", 'w') as outfile:
    json.dump(sent_json, outfile, indent=4)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
