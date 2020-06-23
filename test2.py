from itertools import combinations
sentence = "Attack of the Killer Tomatoes could be enjoyed by any 8yearold with a bad sense of humor, so therefore, it does not qualify as a cult film.There is one good actress in the entire thing Sharon Taylor as Lois Fairchild."

word_list = ['bad', 'good', 'entire']
#


for b in range(len(word_list)):
    number = list(combinations(word_list, b+1))
    for word_tuple in range(len(number)):
        t2l = list(number[word_tuple])
        sentencewords = sentence.split()

        resultwords = [word for word in sentencewords if word.lower() not in t2l]
        result = ' '.join(resultwords)
        print(result)


# query = 'What is hello'
# stopwords = ['what', 'who', 'is', 'a', 'at', 'is', 'he']
# querywords = query.split()
#
# resultwords  = [word for word in querywords if word.lower() not in stopwords]
# result = ' '.join(resultwords)
#
# print(result)