import stanfordnlp
import time

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
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
doc = nlp("It was boring, overdramatic, and the funny parts were too far in between to make up the slack.")
#
doc.sentences[0].print_dependencies()

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# edges = []
# for token in doc.sentences[0].dependencies:
#     if token[0].text.lower() != 'root':
#         edges.append((token[0].text.lower(), token[2].text))
#
# print(edges)

# import stanfordnlp
#
# nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
# doc = nlp("I expected a comedy like the Pretty Mama movies.")
# print(*[f'word: {word.text+" "}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')