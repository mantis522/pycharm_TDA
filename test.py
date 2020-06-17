import json

with open("D:/ruin/data/train_neg_edit_full.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    # print(json_data)

json_string = json_data["splited_sentence"]
json_string2 = json_data['parsed_sentence']

print(json_string2[10000])
print(json_string[10000])