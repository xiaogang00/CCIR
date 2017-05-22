# -*-coding: utf-8 -*-
from sklearn import svm
from get_words_for_CCIR import *
import numpy as np
import gensim
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")


def frequency_word(train_file_dir,train_file_name):
    # 首先需要统计那些词频出现太高的单词，将其进行去除
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    lines = f.readlines()
    word = []
    count = 0
    for line in lines:
        a = line.replace("\n","").split("	")
        if len(a) == 1:
            continue
        if int(a[1]) < 100:
            continue
        word.append(a[0])
        count = count + 1
    return word


def get_vector(word, model):
    try:
        vector = model[word]
    except KeyError:
        vector = np.array([0 for k in range(128)])

    return vector


def get_word_vector(start, end, q, item, word):
    question_matrix = []
    answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)

    # 在这里读入数据，如果word2vec查找不到或者出现的是高频的词的话，就跳过
    for i in range(start, end, 1):
        question_matrix.append([])
        for j in range(len(q[i])):
            temp = get_vector(q[i][j], model)
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
                temp = get_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                answer_matrix[i - start][j].append(temp)

    return question_matrix, answer_matrix


def get_label(start, end):
    train_file_dir = './'
    train_file_name = 'train.1.json'
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    label = []
    count = 0
    for line in f:
        if count < start:
            count = count + 1
            continue
        if count >= end:
            break
        file_dict_temp = json.loads(line)
        passages_list = file_dict_temp['passages']
        label.append([])
        for item in passages_list:
            temp = item['label']
            label[count-start].append(temp)
        count = count + 1
    final_label = []
    for i in range(len(label)):
        for j in range(len(label[i])):
            final_label.append(label[i][j])
    del label
    return final_label


def get_data(start, end, q, item, word):
    question, answer = get_word_vector(start, end, q, item, word)
    question_data = []
    answer_data = []
    # 在这里的形式是，[ [], [], []]，其中的都是问题或者答案的经过word2vec的矩阵
    for i in range(len(question)):
        for j in range(len(answer[i])):
            question_data.append(question[i])

    for i in range(len(answer)):
        for j in range(len(answer[i])):
            answer_data.append(answer[i][j])
    # 获得其label
    label = get_label(start, end)
    del question
    del answer
    question_vector = []
    answer_vector = []
    # 将经过word2vec得到的问题或者答案的矩阵进行逐列平均，得到的是对于每个问题或者答案的向量
    length = 128
    for i in range(len(question_data)):
        question_vector.append([])
        if len(question_data[i]) == 0:
            for number in range(length):
                question_vector[i].append(0)
            continue
        for k in range(length):
            sum = 0
            for m in range(len(question_data[i])):
                sum = sum + question_data[i][m][k]
            mean = sum / len(question_data[i])
            question_vector[i].append(mean)

    for i in range(len(answer_data)):
        answer_vector.append([])
        if len(answer_data[i]) == 0:
            for number in range(length):
                answer_vector[i].append(0)
            continue
        for k in range(length):
            sum = 0
            for m in range(len(answer_data[i])):
                sum = sum + answer_data[i][m][k]
            mean = sum / len(answer_data[i])
            answer_vector[i].append(mean)

    del answer_data, question_data
    # 在这里将其进行问题和答案的拼接，形成最后的q-a对的词频特征向量
    for i in range(len(answer_vector)):
        for j in range(len(question_vector[i])):
            answer_vector[i].append(question_vector[i][j])
    #for i in range(len(answer_vector)):
    #    print len(answer_vector[i])
    return answer_vector, label


def evaluate(q, item, word, clf):
    Total_score_dcg3 = []
    Total_score_dcg5 = []
    for echo in range(4500):
        print 'the echo is', echo
        start = echo
        end = echo + 1
        test_data, test_label = get_data(start, end, q, item, word)
        result = clf.predict(test_data)
        # for i in range(len(test_data)):
        #    print len(test_data[i])
        print result
        print test_label
        temp_score = np.array(result)
        # 在这里需要将最后的label和score结果输入到txt文件当中，便于我们最后的结果分析
        if not os.path.exists('./SVM2'):
            os.makedirs('./SVM2')
        file_object = open('./SVM2/%d' % (echo), 'w')
        for my_number in range(len(temp_score)):
            a = "%d  %lf \n" % (test_label[my_number], temp_score[my_number])
            file_object.write(a)
        print temp_score
        temp_sort = np.argsort(temp_score)
        temp_sort = temp_sort[-1::-1]
        Dcg3 = 0
        Dcg5 = 0
        for number in range(1, 4, 1):
            a = temp_sort[number - 1]
            a = int(a)
            Dcg3 = Dcg3 + (np.power(2, test_label[a]) - 1) / np.log2(number + 1)
        for number in range(1, 6, 1):
            a = temp_sort[number - 1]
            a = int(a)
            Dcg5 = Dcg5 + (np.power(2, test_label[a]) - 1) / np.log2(number + 1)
        Total_score_dcg3.append(Dcg3)
        Total_score_dcg5.append(Dcg5)
        print 'The score for Dcg3 is', np.mean(Total_score_dcg3)
        print 'The score for Dcg5 is', np.mean(Total_score_dcg5)


if __name__ == "__main__":
    train_file_dir = './'
    train_file_name = 'CCIR_train_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)
    train_file_name = 'train.1.json'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    #for echo in range(600):
    #    print 'train echo = ', echo
    #    start = echo*5
    #    end = echo*5 + 5
    start = 0
    end = 4000
    x, label = get_data(start, end, q, item, word)
        # 进行svm的训练
        # clf = svm.SVC()
        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    clf = svm.SVR(kernel='rbf', C=1e2, gamma=0.0001)
    clf.fit(x, label)

    # 进行测试
    evaluate(q, item, word, clf)
