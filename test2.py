from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import re
import pandas as pd
from nltk.tokenize import sent_tokenize

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,parse',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)


    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    def change_moji(self, sentence):
        text = re.sub('\(', '<', sentence)
        text = re.sub('\)', '>', text)
        return text

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

if __name__ == '__main__':
    train_review = []
    train_label = []
    splited_sentence = []
    parsed_sentence = []
    group_list = []
    test = []

    sNLP = StanfordNLP()

    with open("D:/ruin/data/train_neg_full.json") as json_file:
        json_data = json.load(json_file)
        train_data = json_data['data']

    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])

    df_train_data = pd.DataFrame(train_review, columns=['data'])
    df_train_data['label'] = train_label


    for a in df_train_data[9959:9960]['data']:
        split_sentence = sent_tokenize(a)
        splited_sentence.append(split_sentence)


    # print(splited_sentence)

    # for a in range(len(train_data)):
    #     split_sentence = sent_tokenize(train_review[a])
    #     splited_sentence.append(split_sentence)
    #
    with open('tt.json', 'w', encoding='utf-8') as make_file:
        data = {}
        count = 0

        for a in splited_sentence:
            parsed_sentence_lst = []

            for b in a:
                parse1 = " ".join(sNLP.parse(b).split())
                parse1 = sNLP.change_moji(parse1)
                parsed_sentence_lst.append(parse1)
                data['parsed_sentence'] = parsed_sentence_lst
            test.append(parsed_sentence_lst)
            count = count + 1
            print(count)
            # per_20 = count % 20
            # if per_20 == 0:
            #     print(count)

        data['splited_sentence'] = splited_sentence
        data['parsed_sentence'] = test

        json.dump(data, make_file, ensure_ascii=False, indent='\t')
        make_file.close()



