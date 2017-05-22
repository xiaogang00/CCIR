# -*-coding: utf-8 -*-
import numpy as np
from Data2 import *


def get_test_word_vector(start, q, item, word):
    #train_file_dir = './'
    #train_file_name = 'train.1.json'
    #[q, item] = process_train_file(train_file_dir, train_file_name)
    question_matrix = []
    answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)

    for i in range(start, start+5, 1):
        question_matrix.append([])
        count_q = 0
        for j in range(len(q[i])):
            temp = get_vector(q[i][j], model)
            if q[i][j] in word:
                continue
            if np.all(temp == 0):
                continue
            count_q = count_q + 1
            question_matrix[i-start].append(temp)
        #print count_q
 
    for i in range(start, start+5, 1):
        answer_matrix.append([])
        for j in range(len(item[i])):
            count_a = 0
            answer_matrix[i-start].append([])
            #print len(item[i][j])
            for k in range(len(item[i][j])):
                temp = get_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                count_a = count_a + 1
                answer_matrix[i-start][j].append(temp)
            #print count_a

    return question_matrix, answer_matrix


def get_test_data(length_a, length_q, start, q, item, word):
    question, answer = get_test_word_vector(start, q, item, word)
    answer2, question2 = my_padding(question, answer, length_a, length_q)
    question_new = []
    for i in range(len(question2)):
        question_new.append([])
        for j in range(len(answer2[i])):
            question_new[i].append(question2[i])
    del question
    del question2
    height_q = len(question_new[0][0])
    width_q = len(question_new[0][0][0])
    height_a = len(answer2[0][0])
    print len(answer[0][0][0]), length_a, length_q, len(answer2[0][0][0])
    width_a = len(answer2[0][0][0])
    data_question = np.array(question_new)
    data_answer = np.array(answer2)
    question_data = []
    for i in range(len(question_new)):
        data_question[i] = np.array(data_question[i])
        temp = data_question[i].reshape((len(question_new[i]), height_q, width_q, 1))
        question_data.append(temp)

    answer_data = []
    for i in range(len(answer2)):
        data_answer[i] = np.array(data_answer[i])
        temp = data_answer[i].reshape((len(answer2[i]), height_a, width_a, 1))
        answer_data.append(temp)
    return question_data, answer_data


def get_test_label(start):
    train_file_dir = './'
    train_file_name = 'train.1.json'
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    label = []
    count = 0
    for line in f:
        if count < start:
            count = count + 1
            continue
        if count >= start+5:
            break
        file_dict_temp = json.loads(line)
        passages_list = file_dict_temp['passages']
        label.append([])
        for item in passages_list:
            temp = item['label']
            label[count-start].append(temp)
        count = count + 1
    label = np.array(label)
    return label

