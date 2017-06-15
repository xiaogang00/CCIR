from gensim import models
from gensim import corpora
from collections import defaultdict
from pprint import pprint
from matplotlib import pyplot as plt
import os
import logging

def PrintDictionary(dictionary):
    token2id = dictionary.token2id
    dfs = dictionary.dfs
    token_info = {}
    for word in token2id:
        token_info[word] = dict(
            word=word,
            id=token2id[word],
            freq=dfs[token2id[word]]
        )
    token_items = token_info.values()
    token_items = sorted(token_items, key = lambda x:x['id'])
    print('The info of dictionary: ')
    pprint(token_items)
    print('--------------------------')

def Show2dCorpora(corpus):
    nodes = list(corpus)
    ax0 = [x[0][1] for x in nodes]
    ax1 = [x[1][1] for x in nodes]
    plt.plot(ax0, ax1, 'o')
    plt.show()


if __name__ == '__main__':
    documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
    stoplist = set('for a of the end to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token]+=1
    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    PrintDictionary(dictionary)

    tfidf_model = models.TfidfModel(corpus)

    corpus_tfidf = tfidf_model[corpus]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi_model[corpus_tfidf]
    nodes = list(corpus_lsi)
    lsi_model.print_topics(2)
    ax0 = [x[0][1] for x in nodes]
    ax1 = [x[1][1] for x in nodes]
    print(ax0)
    print(ax1)
    plt.plot(ax0, ax1, 'o')
    plt.show()



