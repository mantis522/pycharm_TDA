import time
import json
from nltk.parse.stanford import StanfordParser
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from itertools import combinations

## 시간 존나 오래 걸림 하지말자

start = time.time()


analyser = SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0


os.environ['CLASSPATH'] = "C:/Users/ruin/Desktop/data/stanford-parser-full-2018-10-17"
# os.environ['CLASSPATH'] = '/Users/ruin/Desktop/data/stanford-parser-full-2018-10-17'

stanford_parser = StanfordParser(
    model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
)


with open("C:/Users/ruin/Desktop/data/json_data/TPE_Pattern/EX_neg.json") as json_file:
    json_data = json.load(json_file)

    splited_sentence = json_data['splited_sentence']
    parsed_sentence = json_data['parsed_sentence']
    augmented_sentence = json_data['augmented_text']

# for a in range(10):
#     data = parsed_sentence[a]
#     print(data)

def creating_list(sent, phrase):
    ## 기본적으로 해당 구 찾아주기 위한 함수. 핵심임
    for_remove = []
    parse = stanford_parser.raw_parse(sent)
    CBP = list(parse)
    for sent2 in CBP[0].subtrees():
        if sent2.label() == phrase:
            joined_text = ' '.join(sent2.leaves())
            for_remove.append(joined_text)

    return for_remove

def about_symbol(text):
    text = text.replace(" .", ".")
    text = text.replace(" ?", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(". ", ".")

    return text

# text = "Well what I can say about this movie is that it is great to see so many Asian faces."
#
# test = creating_list(text, "NP")

# for a in test:
#     bb = "Well what I can say about this movie is that it is great to see so many Asian faces."
#     print(a)
#     bb = bb.replace(a, "")
#     bb = " ".join(bb.split())
#     print(bb)

def test(up_temp):
    up_temp = up_temp.replace(about_symbol(d), "")
    up_temp = " ".join(up_temp.split())

    return up_temp

index_num = []

for a in range(len(augmented_sentence)):
    # print(augmented_sentence[a])
    for b in range(len(augmented_sentence[a])):
        if augmented_sentence[a][b] != None:
            index_num.append(a)

index_num = list(set(index_num))
sent_list = []

for a in index_num:
    data = splited_sentence[a]

    for b in range(len(data)):


        list1 = data[:b]
        list2 = data[b + 1:]

        sentence = splited_sentence[a][b]
        word_list = creating_list(sentence, 'PP')

        if len(word_list) > 1:
            first_sent_list = []
            print(str(a + 1) + " 번째 리스트의 " + str(b + 1) + " 번째 문장")
            print("원래 문장 : " + sentence)
            score_original = vader_polarity(sentence)

            print(word_list)

            for c in range(len(word_list)):
                number = list(combinations(word_list, c + 1))
                for word_tuple in range(len(number)):
                    t2l = list(number[word_tuple])
                    # print(t2l)
                    up_temp = sentence
                    for d in t2l:
                        if len(t2l) == 1:
                            under_temp = sentence
                            print("삭제할 구들 : " + about_symbol(d))
                            under_temp = under_temp.replace(about_symbol(d), "")
                            under_temp = " ".join(under_temp.split())
                            print("삭제 이후 문장 : " +under_temp)
                            score_remove = vader_polarity(under_temp)
                            if (score_original == score_remove):
                                result_sentence = ' '.join(list1) + " " + under_temp + " " + ' '.join(list2)
                                # print("해당 문장에 대해서만 감성분석 결과값 : " + str(vader_polarity(result)))
                                first_sent_list.append(result_sentence)
                        else:
                            test_list = []
                            print("삭제할 구들 : " + about_symbol(d))
                            up_temp = up_temp.replace(about_symbol(d), "")
                            up_temp = " ".join(up_temp.split())
                            # print("삭제 이후 문장 : " + up_temp)
                            score_remove = vader_polarity(up_temp)
                            print("두 개 이상 삭제되는 문장 : "+up_temp)
                            ### 리스트로 넣되, 중복은 파기하는 방식으로 해야되나

                            if (score_original == score_remove):
                                result_sentence = ' '.join(list1) + " " + up_temp + " " + ' '.join(list2)
                                # print("해당 문장에 대해서만 감성분석 결과값 : " + str(vader_polarity(result)))
                                first_sent_list.append(result_sentence)


                    first_sent_list = list(set(first_sent_list))
                    # print(first_sent_list)
                    # print("갯수 검사 : "+ str(len(first_sent_list)))
#                     for _ in first_sent_list:
#                         print("중복 검사 : " + _)
#
                    print('----------------------')
# #
            sent_list.append(first_sent_list)
# #
# print(sent_list)

sent_json = {}
sent_json['removed_sentence'] = []
sent_json['removed_sentence'].append(sent_list)

with open("C:/Users/ruin/Desktop/data/json_data/removed_data/removed_PP_neg.json", 'w') as outfile:
    json.dump(sent_json, outfile, indent=4)


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간