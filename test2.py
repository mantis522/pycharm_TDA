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
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

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

    sNLP = StanfordNLP()
    text = 'A blog post using Stanford CoreNLP Server. Visit www.khalidalnajjar.com for more details.'
    parsed_sentence = " ".join(sNLP.parse(text).split())

    with open("C:/Users/ruin/Desktop/data/train_pos_full.json") as json_file:
        json_data = json.load(json_file)
        train_data = json_data['data']

    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])

    df_train_data = pd.DataFrame(train_review, columns=['data'])
    df_train_data['label'] = train_label

    for a in range(20):
        split_sentence = sent_tokenize(train_review[a])
        splited_sentence.append(split_sentence)

    for a in splited_sentence:
        splited_sentence_lst = []
        parsed_sentence_lst = []
        #     print(a)
        splited_sentence_lst = a
        for b in a:
            parse1 = " ".join(sNLP.parse(b).split())
            parse1 = sNLP.change_moji(parse1)
            parsed_sentence_lst.append(parse1)
        print(splited_sentence_lst)
        print(parsed_sentence_lst)
