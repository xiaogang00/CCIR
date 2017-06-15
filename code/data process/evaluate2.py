# -*-coding: utf-8 -*-
import numpy as np
from Data import *


def get_test_word_vector(start, q, item, word):
    # train_file_dir = './'
    # train_file_name = 'train.1.json'
    # [q, item] = process_train_file(train_file_dir, train_file_name)
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

    flag = 0
    final_matrix = []
    for i in range(len(question_new)):
        for j in range(len(question_new[i])):
            if len(question_new[i][j]) != 30 or len(answer2[i][j]) != 320:
                flag = 1
        if flag == 1:
            flag = 0
            temp = np.zeros((len(question_new[i]), 30, 320, 1))
            final_matrix.append(temp)
            continue
        temp_matrix = []
        for k in range(len(question_new[i])):
            if len(answer2[i][j]) != 320:
                temp = np.zeros((30, 320))
                temp_matrix.append(temp)
                continue
            if len(question_new[i][j]) != 30:
                temp = np.zeros((30, 320))
                temp_matrix.append(temp)
                continue
            temp_question = np.array(question_new[i][k])
            temp_answer = np.array(answer2[i][k])
            temp_answer = temp_answer.T
            temp_dot = np.dot(temp_question, temp_answer)
            temp_denominator = temp_dot.copy()
            for mm in range(len(question_new[i][k])):
                for nn in range(len(answer2[i][k])):
                    temp_a = np.array(question_new[i][k][mm])
                    temp_b = np.array(answer2[i][k][nn])
                    La = np.sqrt(temp_a.dot(temp_a))
                    Lb = np.sqrt(temp_b.dot(temp_b))
                    if La == 0 or Lb == 0:
                        temp_denominator[mm, nn] = 0
                    else:
                        temp_denominator[mm, nn] = 1.0 / (La * Lb)
            temp_dot = temp_dot * temp_denominator
            temp_matrix.append(temp_dot)
        temp_matrix = np.array(temp_matrix)
        temp = temp_matrix.reshape((len(question_new[i]), 30, 320, 1))
        final_matrix.append(temp)
    return final_matrix


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


