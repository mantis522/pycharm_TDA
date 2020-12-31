import time
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from itertools import combinations
from nltk import Tree
from tqdm import tqdm

## 삭제할 POS는 SBAR, PP

start = time.time()

analyser = SentimentIntensityAnalyzer()


def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0

file_in = r"D:\data\json_data\TPE_Pattern\popcorn_imdb\ratio\popcorn_TPE_25_neg.json"
file_out = r"D:\data\json_data\removed_data\popcorn\ratio\removed_popcorn_25_PP_neg.json"
deleted_syntax = "PP"

with open(file_in) as json_file:
    json_data = json.load(json_file)

    splited_sentence = json_data['splited_sentence']
    parsed_sentence = json_data['parsed_sentence']
    augmented_sentence = json_data['augmented_text']


# for a in range(10):
#     data = parsed_sentence[a]
#     print(data)

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

index_num = []

for a in range(len(augmented_sentence)):
    # print(augmented_sentence[a])
    for b in range(len(augmented_sentence[a])):
        if augmented_sentence[a][b] != None:
            index_num.append(a)

index_num = list(set(index_num))
sent_list = []
# count = 0

for a in tqdm(index_num):
    data = splited_sentence[a]

    for b in range(len(data)):

        list1 = data[:b]
        list2 = data[b + 1:]

        sentence = splited_sentence[a][b]
        parsed = parsed_sentence[a][b]
        try:
            word_list = creating_list(trans_parsed_sentence(parsed), deleted_syntax)
        except ValueError as e:
            # print("오류 뜨는 문장 : ")
            # print(word_list)
            pass

        if len(word_list) > 1 and len(word_list) < 15:
        # if len(word_list) > 1:
            first_sent_list = []
            print(word_list)
            print(str(a + 1) + " 번째 리스트의 " + str(b + 1) + " 번째 문장")
            print("원래 문장 : " + sentence)
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
                            print("삭제할 구들 : " + about_symbol(d))
                            under_temp = under_temp.replace(about_symbol(d), "")
                            under_temp = " ".join(under_temp.split())
                            print("삭제 이후 문장 : " + under_temp)
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
                    # count = count + 1
                    # print(first_sent_list)
                    # print("갯수 검사 : "+ str(len(first_sent_list)))
                    #                     for _ in first_sent_list:
                    #                         print("중복 검사 : " + _)
                    #
            print('----------------------')
            # #
            sent_list.append(first_sent_list)

        elif len(word_list) > 15:
            word_list = word_list[:15]
            second_sent_list = []
            print(word_list)
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
                            print("삭제 이후 문장 : " + under_temp)
                            score_remove = vader_polarity(under_temp)
                            if (score_original == score_remove):
                                result_sentence = ' '.join(list1) + " " + under_temp + " " + ' '.join(list2)
                                # print("해당 문장에 대해서만 감성분석 결과값 : " + str(vader_polarity(result)))
                                second_sent_list.append(result_sentence)
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
                                second_sent_list.append(result_sentence)

                    second_sent_list = list(set(second_sent_list))
            print('----------------------')
            # #
            sent_list.append(second_sent_list)

    # if count % 10 == 0:
    #     print(count)
# print(sent_list)

sent_json = {}
sent_json['removed_sentence'] = []
sent_json['removed_sentence'].append(sent_list)

with open(file_out, 'w') as outfile:
    json.dump(sent_json, outfile, indent=4)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간