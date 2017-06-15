# -*-coding: utf-8 -*-
from get_words_for_CCIR import *
import numpy as np
import gensim
import sys
import os
from feature import *
reload(sys)
sys.setdefaultencoding("utf-8")


def get_feature_vector(word, model):
    try:
        vector = model[word]
    except KeyError:
        vector = np.array([0 for k in range(128)])

    return vector


def get_feature_word_vector(start, end, q, item, word):
    question_matrix = []
    answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)

    # 在这里读入数据，如果word2vec查找不到或者出现的是高频的词的话，就跳过
    for i in range(start, end, 1):
        question_matrix.append([])
        for j in range(len(q[i])):
            temp = get_feature_vector(q[i][j], model)
            if q[i][j] in word:
                continue
            if np.all(temp == 0):
                continue
            question_matrix[i - start].append(temp)

    for i in range(start, end, 1):
        answer_matrix.append([])
        for j in range(len(item[i])):
            answer_matrix[i - start].append([])
            for k in range(len(item[i][j])):
                temp = get_feature_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                answer_matrix[i - start][j].append(temp)

    return question_matrix, answer_matrix


def get_feature_data(start, end, q, item, word):
    train_file_dir = './'
    train_file_name = 'CCIR_train_3_word_num.txt'
    word2 = frequency_statistical(train_file_dir, train_file_name)
    question, answer = get_feature_word_vector(start, end, q, item, word)
    question_data = []
    answer_data = []
    for i in range(len(question)):
        for j in range(len(answer[i])):
            question_data.append(question[i])

    for i in range(len(answer)):
        for j in range(len(answer[i])):
            answer_data.append(answer[i][j])
    # 在这里的形式是，[ [], [], []]，其中的都是问题或者答案的经过word2vec的矩阵
    feature = similarity_feature(question_data, answer_data)
    # feature是用来衡量问题与答案之间相似度的特征，有16个特征
    # feature2 = metadata_feature(question, answer)
    # feature2有4个特征
    feature3 = frequency_feature(start, end, q, item, word2)
    # feature3有4个特征
    feature4 = same_word_feature(question_data, answer_data)
    # feature4有4个特征
    del answer_data, question_data
    del answer, question
    # 在这里将其进行问题和答案的拼接，形成最后的q-a对的词频特征向量
    for i in range(len(feature)):
        #for j in range(len(feature2[i])):
        #    feature[i].append(feature2[i][j])
        for n in range(len(feature3[i])):
            feature[i].append(feature3[i][n])
        for mm in range(len(feature4[i])):
            feature[i].append(feature4[i][mm])

    for i in range(len(feature)):
        for j in range(len(feature[i])):
            if np.isnan(feature[i][j]):
                feature[i][j] = 0
        max_item = max(feature[i])
        min_item = min(feature[i])
        if max_item == min_item:
            continue
        for k in range(len(feature[i])):
            feature[i][k] = (feature[i][k] - min_item) / (max_item - min_item)
    #feature = np.array(feature)
    #feature = feature.reshape((len(feature), len(feature[0]), 1))
    return feature