import json
import random

file_name = r"D:\data\json_data\parsed_data\popcorn_imdb\ratio\popcorn_parsed_50_pos.json"

with open(r"D:\data\json_data\parsed_data\popcorn_imdb\popcorn_imdb_parsed_pos.json") as json_file:
    json_data = json.load(json_file)

    splited_sentence = json_data['splited_sentence']
    parsed_sentence = json_data['parsed_sentence']

parsed_sen_list = []

sent_json = {}

def making_list(ratio):
    len_of_sen = len(parsed_sentence)
    ra = int(len_of_sen) / 100 * ratio
    ra = int(ra)

    parsed_split = parsed_sentence[:ra]
    sent_split = splited_sentence[:ra]

    sent_json['splited_sentence'] = []
    sent_json['parsed_sentence'] = []
    sent_json['splited_sentence'].append(sent_split)
    sent_json['parsed_sentence'].append(parsed_split)

    with open(file_name, 'w') as out_file:
        json.dump(sent_json, out_file, indent=4)

making_list(50)