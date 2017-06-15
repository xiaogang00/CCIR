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
    height_q = len(question_new[0][0])
    if height_q != 30:
        height_q = 30
    width_q = len(question_new[0][0][0])
    height_a = len(answer2[0][0])
    if height_a != 320:
        height_a = 320
    width_a = len(answer2[0][0][0])
    final_question = []
    final_answer = []
    for i in range(len(question_new)):
        for j in range(len(question_new[i])):
            if len(question_new[i][j]) != 30:
                temp = np.zeros((height_q, width_q, 1))
                final_question.append(temp)
                continue
            temp1 = np.array(question_new[i][j])
            temp2 = temp1.reshape(height_q, width_q, 1)
            final_question.append(temp2)

    for i in range(len(answer2)):
        for j in range(len(answer2[i])):
            if len(answer2[i][j]) != 320:
                temp = np.zeros((height_a, width_a, 1))
                final_answer.append(temp)
                continue
            temp1 = np.array(answer2[i][j])
            temp2 = temp1.reshape(height_a, width_a, 1)
            final_answer.append(temp2)
    del question_new
    del answer2
    final_answer = np.array(final_answer)
    final_question = np.array(final_question)
    return final_question, final_answer, height_a, height_q, width_a, width_q


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




