import json

with open("C:/Users/ruin/Desktop/data/edited_data/py_train_pos_edit_full.json", encoding='utf-8') as json_file:
  json_data = json.load(json_file)
  parsed_sentence = json_data['parsed_sentence']
  print(parsed_sentence[0][0])