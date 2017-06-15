# -*-coding: utf-8 -*-
from get_words_for_CCIR import *
import gensim
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


def loaddata():
    file = open('train.1.json', 'r')
    count = 0
    for line in file.readlines():
        count = count + 1
        dic = json.loads(line)
        f = dic['query']
        print f
        length = len(dic['passages'])
        if count == 1:
            for i in range(length):
                answer = dic['passages'][i]['passage_text']
                label = dic['passages'][i]['label']
                print answer, label, '\n'


def my_padding(data_q, data_a, length_a, length_q):
    if length_q != 0 and length_a != 0:
        for i in range(len(data_q)):
            m = len(data_q[i])
            a = int((length_q - m) / 2)
            b = length_q - m - a
            for j in range(a):
                temp = np.array([0 for k in range(128)])
                data_q[i].append(temp)
            for j in range(b):
                temp = np.array([0 for k in range(128)])
                data_q[i].insert(0, temp)

        for i in range(len(data_a)):
            for j in range(len(data_a[i])):
                m = len(data_a[i][j])
                a = int((length_a - m)) / 2
                b = length_a - m - a
                for number in range(a):
                    temp = np.array([0 for k in range(128)])
                    data_a[i][j].append(temp)
                for number in range(b):
                    temp = np.array([0 for k in range(128)])
                    data_a[i][j].insert(0, temp)
    return data_a, data_q


def find_max(q, item, word):
    # train_file_dir = './'
    # train_file_name = 'train.1.json'
    # [q, item] = process_train_file(train_file_dir, train_file_name)
    # question_matrix = []
    # answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)
    len_q = []
    len_a = []
    for i in range(len(q)):
        count = 0
        for j in range(len(q[i])):
            temp = get_vector(q[i][j], model)
            if q[i][j] in word:
                continue
            if np.all(temp == 0):
                continue
            count = count + 1
        len_q.append(count)

    for i in range(len(item)):
        len_a.append([])
        for j in range(len(item[i])):
            count = 0
            for k in range(len(item[i][j])):
                temp = get_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                count = count + 1
            len_a[i].append(count)

    length_q = 30
    length_a = 30
    for i in range(len(len_q)):
        m = len_q[i]
        if m > length_q:
            length_q = m

    for i in range(len(len_a)):
        for j in range(len(len_a[i])):
            m = len_a[i][j]
            if m > length_a:
                length_a = m

    return length_a, length_q


def get_vector(word, model):
    try:
        vector = model[word]
    except KeyError:
        vector = np.array([0 for k in range(128)])

    return vector


def get_word_vector(start, q, item, word):
    # train_file_dir = './'
    # train_file_name = 'train.1.json'
    # [q, item] = process_train_file(train_file_dir, train_file_name)
    question_matrix = []
    answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)

    for i in range(start, start + 5, 1):
        question_matrix.append([])
        for j in range(len(q[i])):
            temp = get_vector(q[i][j], model)
            if q[i][j] in word:
                continue
            if np.all(temp == 0):
                continue
            question_matrix[i - start].append(temp)

    for i in range(start, start + 5, 1):
        answer_matrix.append([])
        for j in range(len(item[i])):
            answer_matrix[i - start].append([])
            for k in range(len(item[i][j])):
                temp = get_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                answer_matrix[i - start][j].append(temp)

    return question_matrix, answer_matrix


def get_train_data(start, lengtha, lengthq, q, item, word):
    question, answer = get_word_vector(start, q, item, word)
    answer2, question2 = my_padding(question, answer, lengtha, lengthq)
    question_new = []
    for i in range(len(question2)):
        question_new.append([])
        for j in range(len(answer2[i])):
            question_new[i].append(question2[i])

    del question
    del answer
    del question2
    final_matrix = []
    for i in range(len(question_new)):
        for j in range(len(question_new[i])):
            if len(answer2[i][j]) != 320:
                temp = np.zeros((30, 320, 1))
                final_matrix.append(temp)
                continue
            if len(question_new[i][j]) != 30:
                temp = np.zeros((30, 320, 1))
                final_matrix.append(temp)
                continue
            temp_question = question_new[i][j]
            temp_answer = answer2[i][j]
            temp_question = np.array(temp_question)
            temp_answer = np.array(temp_answer)
            temp_answer = temp_answer.T
            temp = np.dot(temp_question, temp_answer)
            temp_denominator = temp.copy()
            for mm in range(len(question_new[i][j])):
                for nn in range(len(answer2[i][j])):
                    temp_a = np.array(question_new[i][j][mm])
                    temp_b = np.array(answer2[i][j][nn])
                    La = np.sqrt(temp_a.dot(temp_a))
                    Lb = np.sqrt(temp_b.dot(temp_b))
                    if La == 0 or Lb == 0:
                        temp_denominator[mm, nn] = 0
                    else:
                        temp_denominator[mm, nn] = 1.0 / (La * Lb)
            temp = temp * temp_denominator
            temp = temp.reshape(30, 320, 1)
            final_matrix.append(temp)
    final_matrix = np.array(final_matrix)
    del question_new
    del answer2
    return final_matrix


def get_label(start):
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
            if temp == 0:
                temp = 0
            elif temp == 1:
                temp = 1
            elif temp == 2:
                temp = 2
            label[count - start].append(temp)
        count = count + 1
    final_label = []
    for i in range(len(label)):
        for j in range(len(label[i])):
            final_label.append(label[i][j])
    del label
    final_label = np.array(final_label)
    return final_label


if __name__ == '__main__':
    get_train_data()
    # question, answer = get_word_vector()
    # label = get_label()
    # print label




