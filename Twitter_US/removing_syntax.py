import time
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from itertools import combinations
from nltk import Tree
from tqdm import tqdm

## amod는 삭제 가능한 형용사 제거를 위해 따로 dependency parsing 하느라 새로 만든거임

start = time.time()

analyser = SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    dic_max = max(score.values())
    if dic_max == score['neg']:
        return 0
    elif dic_max == score['neu']:
        return 1
    elif dic_max == score['pos']:
        return 2
    else: ## compound의 경우.
        return 3


with open("D:/data/json_data/parsed_data/tweet_us_parsed_neg.json") as json_file:
    json_data = json.load(json_file)

    splited_sentence = json_data['splited_sentence']
    parsed_sentence = json_data['parsed_sentence']

def creating_list(sent, phrase):
    for_remove = []
    tree = Tree.fromstring(trans_parsed_sentence(sent))
    for sent in tree[0].subtrees():
        if sent.label() == phrase:
            joined_text = ' '.join(sent.leaves())
            for_remove.append(joined_text)

    return for_remove


def about_symbol(text):
    text = text.replace(" .", ".")
    text = text.replace(" ?", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(". ", ".")

    return text


def trans_parsed_sentence(text):
    text = text.replace("<", "(")
    text = text.replace(">", ")")

    return text

sent_list = []

count = 0
for a in tqdm(range(len(splited_sentence))):
    data = splited_sentence[a]

    for b in range(len(data)):

        list1 = data[:b]
        list2 = data[b + 1:]

        sentence = splited_sentence[a][b]
        parsed = parsed_sentence[a][b]
        try:
            word_list = creating_list(trans_parsed_sentence(parsed), 'PP')
        except ValueError as e:
            print("오류 뜨는 문장 : ")
            print(word_list)
            pass

        if len(word_list) > 1 and len(word_list) < 13:
        # if len(word_list) > 1:
            first_sent_list = []
            print(word_list)
            print(str(a + 1) + " 번째 리스트의 " + str(b + 1) + " 번째 문장")
            print(count)
            # print("원래 문장 : " + sentence)
            score_original = vader_polarity(sentence)

            # print(word_list)

            for c in range(len(word_list)):
                number = list(combinations(word_list, c + 1))
                for word_tuple in range(len(number)):
                    t2l = list(number[word_tuple])
                    # print(t2l)
                    up_temp = sentence
                    for d in t2l:
                        if len(t2l) == 1:
                            under_temp = sentence
                            # print("삭제할 구들 : " + about_symbol(d))
                            under_temp = under_temp.replace(about_symbol(d), "")
                            under_temp = " ".join(under_temp.split())
                            # print("삭제 이후 문장 : " + under_temp)
                            score_remove = vader_polarity(under_temp)
                            if (score_original == score_remove):
                                result_sentence = ' '.join(list1) + " " + under_temp + " " + ' '.join(list2)
                                # print("해당 문장에 대해서만 감성분석 결과값 : " + str(vader_polarity(result)))
                                first_sent_list.append(result_sentence)
                        else:
                            test_list = []
                            # print("삭제할 구들 : " + about_symbol(d))
                            up_temp = up_temp.replace(about_symbol(d), "")
                            up_temp = " ".join(up_temp.split())
                            score_remove = vader_polarity(up_temp)
                            # print("두 개 이상 삭제되는 문장 : " + up_temp)
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
            count = count + 1

    # if count % 10 == 0:
    #     print(count)
# print(sent_list)

sent_json = {}
sent_json['removed_sentence'] = []
sent_json['removed_sentence'].append(sent_list)

with open("D:/data/json_data/removed_data/Twitter_US_Airline/removed_PP_neg.json", 'w', encoding='utf-8') as outfile:
    json.dump(sent_json, outfile, indent=4)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간