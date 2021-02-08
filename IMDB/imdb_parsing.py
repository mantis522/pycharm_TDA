import json
import pandas as pd
import time
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm
import re
import contractions
import stanfordnlp

start = time.time()

dir_name_pos = r"D:\data\json_data\parsed_data\imdb\imdb_parsed_pos_5.json"
dir_name_neg = r"D:\data\json_data\parsed_data\imdb\imdb_parsed_neg_5.json"

def making_parsed_tree(sentiment_code, file_name):
    splited_sentence_first = []
    parsed_sentence_first = []

    pcn = StanfordCoreNLP('http://localhost:9000')
    df = pd.read_csv(r"D:\data\csv_file\imdb\imdb_5.csv")

    nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
    text = df['review']
    label = df['label']
    del df['Unnamed: 0']

    sent_json = {}

    def about_symbol(text):
        text = re.sub("@", '', text)
        text = re.sub(r'http\S+', '', text)
        return text

    for a in tqdm(range(len(df))):
        tweet_txt = about_symbol(text[a])
        if label[a] == sentiment_code:
            if len(tweet_txt) > 3:
                tweet_txt = " ".join(tweet_txt.split())
                tweet_txt = contractions.fix(tweet_txt)

                doc = nlp(tweet_txt)
                splited_sentence_second = []
                parsed_sentence_second = []

                for sentence in doc.sentences:
                    temp = []
                    for token in sentence.tokens:
                        temp.append(token.text)
                    sum_text = " ".join(temp)
                    sum_text = about_symbol(sum_text)
                    output = pcn.annotate(sum_text, properties={
                        'annotators': 'parse',
                        'outputFormat': 'json'
                    })
                    parsed_sent = output['sentences'][0]['parse']
                    parsed_sent = " ".join(parsed_sent.split())
                    parsed_sent = parsed_sent.replace('(', '<')
                    parsed_sent = parsed_sent.replace(')', '>')

                    parsed_sentence_second.append(parsed_sent)
                    splited_sentence_second.append(sum_text)
                    # print(parsed_sent)
                splited_sentence_first.append(splited_sentence_second)
                parsed_sentence_first.append(parsed_sentence_second)

            sent_json['splited_sentence'] = []
            sent_json['parsed_sentence'] = []
            sent_json['original_sentence'] = []
            sent_json['splited_sentence'].append(splited_sentence_first)
            sent_json['parsed_sentence'].append(parsed_sentence_first)
            sent_json['original_sentence'].append(tweet_txt)

    with open(file_name, 'w') as out_file:
        json.dump(sent_json, out_file, indent=4)

making_parsed_tree(1, dir_name_pos)
making_parsed_tree(0, dir_name_neg)

print("time :", time.time() - start)