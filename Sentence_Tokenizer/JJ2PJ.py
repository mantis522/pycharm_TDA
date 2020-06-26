import json
import stanfordnlp
import time
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

start = time.time()

pcn = StanfordCoreNLP('http://localhost:9000')

splited_sentence_first = []
parsed_sentence_first = []
nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')


def about_symbol(text):
    text = text.replace(" .", ".")
    text = text.replace(" ?", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(". ", ".")

    return text

with open("C:/Users/ruin/Desktop/data/json_data/train_neg_full.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    data_list = json_data['data']
    # print(data_list[0]['txt'])

for a in tqdm(range(len(data_list))):
    json_txt = data_list[a]['txt']
    # print(json_txt)
    doc = nlp(json_txt)
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
    # count = count + 1
    #
    # if count % 10 == 0:
    #     print(count)

sent_json = {}
sent_json['splited_sentence'] = []
sent_json['parsed_sentence'] = []
sent_json['original_sentence'] = []
sent_json['splited_sentence'].append(splited_sentence_first)
sent_json['parsed_sentence'].append(parsed_sentence_first)
sent_json['original_sentence'].append(data_list)
#
with open("C:/Users/ruin/Desktop/data/edited_data/py_train_neg_edit_full.json", 'w') as out_file:
    json.dump(sent_json, out_file, indent=4)


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간