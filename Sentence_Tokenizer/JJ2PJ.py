import json
import stanfordnlp

splited_sentence_first = []
nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')

def about_symbol(text):
    text = text.replace(" .", ".")
    text = text.replace(" ?", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(". ", ".")

    return text

with open("C:/Users/ruin/Desktop/data/json_data/train_pos_full.json", encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    data_list = json_data['data']
    # print(data_list[0]['txt'])


# for a in range(len(data_list)):
#     splited_sentence_second = []
#     json_txt = data_list[a]['txt']
#     for senten

txt = data_list[0]['txt']
doc = nlp(txt)

for sentence in doc.sentences:
    test = []
    for token in sentence.tokens:
        test.append(token.text)

    sum_text = " ".join(test)
    sum_text = about_symbol(sum_text)

    print(sum_text)