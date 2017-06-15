# -*-coding: utf-8 -*-
from sklearn import svm
from get_words_for_CCIR import *
import numpy as np
import gensim
import sys
import os
import pickle
from feature import *
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


def get_svm_vector(word, model):
    try:
        vector = model[word]
    except KeyError:
        vector = np.array([0 for k in range(128)])

    return vector


def get_svm_word_vector(start, end, q, item, word):
    question_matrix = []
    answer_matrix = []
    gensim_model = "wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)

    # 在这里读入数据，如果word2vec查找不到或者出现的是高频的词的话，就跳过
    for i in range(start, end, 1):
        question_matrix.append([])
        for j in range(len(q[i])):
            temp = get_svm_vector(q[i][j], model)
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
                temp = get_svm_vector(item[i][j][k], model)
                if item[i][j][k] in word:
                    continue
                if np.all(temp == 0):
                    continue
                answer_matrix[i - start][j].append(temp)

    return question_matrix, answer_matrix


def get_svm_label(start, end):
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'
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


def get_svm_data(start, end, q, item, word):
    train_file_dir = './'
    train_file_name = 'CCIR_train_3_word_num.txt'
    word2 = frequency_statistical(train_file_dir, train_file_name)
    question, answer = get_svm_word_vector(start, end, q, item, word)
    question_data = []
    answer_data = []
    # 在这里的形式是，[ [], [], []]，其中的都是问题或者答案的经过word2vec的矩阵
    for i in range(len(question)):
        for j in range(len(answer[i])):
            question_data.append(question[i])

    for i in range(len(answer)):
        for j in range(len(answer[i])):
            answer_data.append(answer[i][j])
    feature = similarity_feature(question_data, answer_data)
    feature2 = metadata_feature(question, answer)
    feature3 = frequency_feature(start, end, q, item, word2)
    feature4 = same_word_feature(question_data, answer_data)
    # 获得其label
    #label = get_label(start, end)
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
    for i in range(len(feature)):
        for j in range(len(feature2[i])):
            feature[i].append(feature2[i][j])
        for n in range(len(feature3[i])):
            feature[i].append(feature3[i][n])
        for mm in range(len(feature4[i])):
            feature[i].append(feature4[i][mm])

    del feature2, feature3, feature4
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

    for i in range(len(answer_vector)):
        for j in range(len(question_vector[i])):
            answer_vector[i].append(question_vector[i][j])
        for k in range(len(feature[i])):
            answer_vector[i].append(feature[i][k])
        # for m in range(len(feature2[i])):
        #    answer_vector[i].append(feature2[i][m])
        # for n in range(len(feature3[i])):
        #    answer_vector[i].append(feature3[i][n])
        # for mm in range(len(feature4[i])):
        #    answer_vector[i].append(feature4[i][mm])

    # for i in range(len(answer_vector)):
    #    for j in range(len(answer_vector[i])):
    #        if np.isnan(answer_vector[i][j]):
    #            answer_vector[i][j] = 0
    return answer_vector


def evaluate(q, item, word, clf1):
    Total_score_dcg3 = []
    Total_score_dcg5 = []
    Best_score_dcg3 = []
    Best_score_dcg5 = []
    for echo in range(7998):
        print 'the echo is', echo
        start = echo
        end = echo + 5
        test_data = get_svm_data(start, end, q, item, word)
        test_label = get_svm_label(start, end)
        # print len(test_data)
        label_len = []
        for number3 in range(5):
            test_label1 = get_svm_label(start + number3, start + number3 + 1)
            label_len.append(len(test_label1))

        result1 = clf1.predict(test_data)
        # result2 = clf2.predict(test_data)
        result = np.zeros((len(result1), 1))
        for i in range(len(result1)):
            result[i] = (result1[i])

        count = 0
        for number4 in range(5):
            result_temp = []
            label_temp = []
            for number5 in range(label_len[number4]):
                result_temp.append(result[count])
                label_temp.append(test_label[count])
                count += 1
            temp_score = np.array(result_temp)
            # 在这里需要将最后的label和score结果输入到txt文件当中，便于我们最后的结果分析
            if not os.path.exists('./SVM3'):
                os.makedirs('./SVM3')
            file_object = open('./SVM3/%d' % (echo*5 + number4), 'w')
            for my_number in range(len(temp_score)):
                a = "%d  %lf \n" % (label_temp[my_number], temp_score[my_number])
                file_object.write(a)
            print temp_score
            print label_temp
            temp_sort = np.argsort(temp_score)
            temp_sort = temp_sort[-1::-1]
            Dcg3 = 0
            Dcg5 = 0
            for number in range(1, 4, 1):
                a = temp_sort[number - 1]
                a = int(a)
                Dcg3 = Dcg3 + (np.power(2, label_temp[a]) - 1) / np.log2(number + 1)
            for number in range(1, 6, 1):
                a = temp_sort[number - 1]
                a = int(a)
                Dcg5 = Dcg5 + (np.power(2, label_temp[a]) - 1) / np.log2(number + 1)
            Total_score_dcg3.append(Dcg3)
            Total_score_dcg5.append(Dcg5)

            best_label = np.array(label_temp)
            temp_sort2 = np.argsort(best_label)
            temp_sort2 = temp_sort2[-1::-1]
            Best_Dcg3 = 0
            Best_Dcg5 = 0
            for number in range(1, 4, 1):
                a = temp_sort2[number - 1]
                a = int(a)
                Best_Dcg3 = Best_Dcg3 + (np.power(2, label_temp[a]) - 1) / np.log2(number + 1)
            for number in range(1, 6, 1):
                a = temp_sort2[number - 1]
                a = int(a)
                Best_Dcg5 = Best_Dcg5 + (np.power(2, label_temp[a]) - 1) / np.log2(number + 1)
            Best_score_dcg3.append(Best_Dcg3)
            Best_score_dcg5.append(Best_Dcg5)
        print 'The score for Dcg3 is', np.mean(Total_score_dcg3)
        print 'The score for Dcg5 is', np.mean(Total_score_dcg5)
        print 'The best score for Dcg3 is', np.mean(Best_score_dcg3)
        print 'The best score for Dcg5 is', np.mean(Best_score_dcg5)



if __name__ == "__main__":
    train_file_dir = './'
    train_file_name = 'CCIR_train_3_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)
    train_file_name = 'json.longshort.new.all'
    #[q, item] = process_train_file(train_file_dir, train_file_name)
    f3 = file('q.pkl', 'rb')
    f4 = file('item.pkl', 'rb')
    q = pickle.load(f3)
    item = pickle.load(f4)
    f3.close()
    f4.close()
    #f1 = file('x.pkl', 'wb')
    #f2 = file('label.pkl', 'wb')
    #pickle.dump(x, f1, True)
    #pickle.dump(label, f2, True)
    #f1.close()
    #f2.close()

    start = 0
    end = 4000
    x = get_svm_data(start, end, q, item, word)
    label = get_svm_label(start, end)
    clf1 = svm.SVR(kernel='rbf', C=1e2, gamma=0.0001)
    clf1.fit(x, label)

    # 进行测试
    evaluate(q, item, word, clf1)
