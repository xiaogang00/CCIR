# -*-coding: utf-8 -*-
import numpy as np
from Data2 import *

# 这是用CNN模型来进行测试的代码。其中读入数据的方法和data2.py中一样，start是当前读入的数据
# 在当前存储的q,item,qaid中的index。这里不处理无法分词的句子。

def get_test_word_vector(start, q, item, word):
    question_matrix = []
    answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)

    for i in range(start, start + 5, 1):
        question_matrix.append([])
        count_q = 0
        for j in range(len(q[i])):
            temp = get_vector(q[i][j], model)
            if q[i][j] in word:
                continue
            if np.all(temp == 0):
                continue
            count_q = count_q + 1
            question_matrix[i - start].append(temp)
            # print count_q

    for i in range(start, start + 5, 1):
        answer_matrix.append([])
        for j in range(len(item[i])):
            count_a = 0
            answer_matrix[i - start].append([])
            # print len(item[i][j])
            for k in range(len(item[i][j])):
                temp = get_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                count_a = count_a + 1
                answer_matrix[i - start][j].append(temp)
                # print count_a

    return question_matrix, answer_matrix


def getid(start, qaid):
    qa_maxrix = []
    for i in range(start, start + 5, 1):
        qa_maxrix.append(qaid[i])
    return qa_maxrix


def get_test_data(length_a, length_q, start, q, item, qaid, word):
    question, answer = get_test_word_vector(start, q, item, word)
    answer2, question2 = my_padding(question, answer, length_a, length_q)
    qa_matrix = getid(start, qaid)
    question_new = []
    for i in range(len(question2)):
        question_new.append([])
        for j in range(len(answer2[i])):
            question_new[i].append(question2[i])
    del question
    del question2
    height_q = len(question_new[0][0])
    if height_q != 30:
        height_q = 30
    width_q = len(question_new[0][0][0])
    height_a = len(answer2[0][0])
    if height_a != 320:
        height_a = 320
    width_a = len(answer2[0][0][0])
    data_question = np.array(question_new)
    data_answer = np.array(answer2)
    question_data = []
    flag = 0
    for i in range(len(question_new)):
        for j in range(len(question_new[i])):
            if len(question_new[i][j]) != 30:
                flag = 1
        if flag == 1:
            flag = 0
            temp = np.zeros((len(question_new[i]), height_q, width_q, 1))
            question_data.append(temp)
            continue
        temp_data_q = np.array(question_new[i])
        temp = temp_data_q.reshape((len(question_new[i]), height_q, width_q, 1))
        question_data.append(temp)

    answer_data = []
    flag2 = 0
    for i in range(len(answer2)):
        for j in range(len(answer2[i])):
            if len(answer2[i][j]) != 320:
                flag2 = 1
        if flag2 == 1:
            flag2 = 0
            temp = np.zeros((len(answer2[i]), height_a, width_a, 1))
            answer_data.append(temp)
            continue
        temp_data = np.array(answer2[i])
        temp = temp_data.reshape((len(answer2[i]), height_a, width_a, 1))
        answer_data.append(temp)
    return question_data, answer_data, qa_matrix


def get_test_label(start):
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    label = []
    count = 0
    for line in f:
        if count < start:
            count = count + 1
            continue
        if count >= start + 5:
            break
        file_dict_temp = json.loads(line)
        passages_list = file_dict_temp['passages']
        label.append([])
        for item in passages_list:
            temp = item['label']
            label[count - start].append(temp)
        count = count + 1
    label = np.array(label)
    return label

