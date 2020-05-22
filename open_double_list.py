import json


text_list = []
text_data = {}

with open("C:/Users/ruin/Desktop/data/data_augmentation/imi_2.json") as json_file:
    json_data = json.load(json_file)
    count = 0
    json_string = json_data["augmented_text"]
    for a in json_string:
        # print(' '.join(a))
        text = ' '.join(a)
        text_list.append(text)
    json_file.close()

text_data['augmented_text'] = text_list
# print(text_data)
with open("test.json", 'w', encoding='utf-8') as make_file:
    json.dump(text_data, make_file, ensure_ascii=False, indent='\t')
    make_file.close()