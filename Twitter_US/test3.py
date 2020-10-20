import json
from tqdm import tqdm


with open("D:/data/json_data/parsed_data/tweet_us_parsed_neg.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    json_string = json_data['splited_sentence']

for a in tqdm(range(len(json_string))):
    print(a)
