import stanfordnlp
import time
import re
from itertools import combinations

start = time.time()
#
# # config = {
# #         'processors': 'tokenize,pos',
# #         'tokenize_pretokenized': True,
# #         'pos_model_path': "C:/Users/ruin/stanfordnlp_resources/en_ewt_models/en_ewt_tagger.pt",
# #         'pos_pretrain_path': "C:/Users/ruin/stanfordnlp_resources/en_ewt_models/en_ewt.pretrain.pt",
# #         'pos_batch_size': 1000
# #          }
# # nlp = stanfordnlp.Pipeline(**config)
#
nlp = stanfordnlp.Pipeline(processors='tokenize,pos,depparse') # This sets up a default neural pipeline in English
sentence = "Attack of the Killer Tomatoes could be enjoyed by any 8yearold with a bad sense of humor, so therefore, it does not qualify as a cult film.There is one good actress in the entire thing Sharon Taylor as Lois Fairchild."
doc = nlp(sentence)
word_list = []

for i, _ in enumerate(doc.sentences):
    for word in doc.sentences[i].words:
        if word.dependency_relation == 'amod':
            word_list.append(word.text)


print(word_list)

for b in range(len(word_list)):
    number = list(combinations(word_list, b+1))
    for word_tuple in range(len(number)):
        t2l = list(number[word_tuple])
        sentencewords = sentence.split()

        resultwords = [word for word in sentencewords if word.lower() not in t2l]
        result = ' '.join(resultwords)
        print(result)


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간