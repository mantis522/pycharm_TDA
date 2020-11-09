import json
import time
import stanfordnlp
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from itertools import combinations
from nltk import Tree
from tqdm import tqdm

## 삭제할 POS는 SBAR, PP

start = time.time()

analyser = SentimentIntensityAnalyzer()
nlp = stanfordnlp.Pipeline(processors='tokenize,pos,depparse')

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0


with open("D:/data/json_data/TPE_Pattern/amazon/amazon_EX_100000_pos.json") as json_file:
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
        word_list = []
        list1 = data[:b]
        list2 = data[b + 1:]
        sentence = splited_sentence[a][b]

        doc = nlp(sentence)

        for i, _ in enumerate(doc.sentences):
            for word in doc.sentences[i].words:
                if word.dependency_relation == 'amod':
                    word_list.append(word.text)

        if len(word_list) > 1:
            first_sent_list = []
            # print(str(a + 1) + " 번째 리스트의 " + str(b + 1) + " 번째 문장")
            score_original = vader_polarity(sentence)

            # print(word_list)

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
                        # print(result_sentence)
                        first_sent_list.append(result_sentence)

                    # print("전체 세그먼트에 대한 감성분석 결과값 : " + str(vader_polarity(result_sentence)))

            sent_list.append(first_sent_list)

    # if count % 10 == 0:
    #     print(count)
# print(sent_list)

sent_json = {}
sent_json['removed_sentence'] = []
sent_json['removed_sentence'].append(sent_list)

with open("D:/data/json_data/removed_data/amazon/removed_amod_pos_100000.json", 'w') as outfile:
    json.dump(sent_json, outfile, indent=4)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간