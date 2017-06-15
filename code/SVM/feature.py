# -*-coding: utf-8 -*-
import numpy as np
import pickle
import numpy.linalg as LA
from svm_practice3 import *

def similarity_feature(question, answer):
    # centroid vector
    # 先将word2vec的数据变成向量的形式
    question_vector = []
    for i in range(len(question)):
        question_vector.append([])
        if len(question[i]) == 0:
            for number in range(128):
                question_vector[i].append(0)
            continue
        for k in range(128):
            sum = 0
            for m in range(len(question[i])):
                sum = sum + question[i][m][k]
            mean = sum / len(question[i])
            question_vector[i].append(mean)

    answer_vector = []
    for i in range(len(answer)):
        answer_vector.append([])
        if len(answer[i]) == 0:
            for number in range(128):
                answer_vector[i].append(0)
            continue
        for k in range(128):
            sum = 0
            for m in range(len(answer[i])):
                sum = sum + answer[i][m][k]
            mean = sum / len(answer[i])
            answer_vector[i].append(mean)

    feature = []
    for i in range(len(answer)):
        feature.append([])
        temp_question_vector = np.array(question_vector[i])
        temp_answer_vector = np.array(answer_vector[i])
        dot1 = temp_question_vector.dot(temp_answer_vector)
        #dot = np.multiply(temp_question_vector, temp_answer_vector)
        Lq = np.sqrt(temp_question_vector.dot(temp_question_vector))
        La = np.sqrt(temp_answer_vector.dot(temp_answer_vector))
        if La == 0 or Lq == 0:
            temp = 0
        else:
            temp = dot1 / (Lq * La)
        # 在这里加入的相似度特征是余弦相似，欧氏距离的范数
        feature[i].append(temp)
        distance = temp_question_vector - temp_answer_vector
        feature[i].append(LA.norm(distance, 1))
        feature[i].append(LA.norm(distance, 2))
        feature[i].append(LA.norm(distance, np.inf))

        # 下面是衡量答案中的单词与问题之间的相似度，相似度最大的可以作为其feature
        rank_similarity = []
        distance_similarity = []
        distance_similarity2 = []
        distance_similarity3 = []
        for j in range(len(answer[i])):
            temp = np.array(answer[i][j])
            dot2 = temp.dot(temp_question_vector)
            Lq = np.sqrt(temp_question_vector.dot(temp_question_vector))
            La = np.sqrt(temp.dot(temp))
            if Lq == 0 or La == 0:
                s = 0
            else:
                s = dot2 / (Lq * La)
            rank_similarity.append(s)
            distance2 = temp - temp_question_vector
            distance_similarity.append(LA.norm(distance2, 1))
            distance_similarity2.append(LA.norm(distance2, 2))
            distance_similarity3.append(LA.norm(distance2, np.inf))

        for number2 in range(len(rank_similarity)):
            if np.isnan(rank_similarity[number2]):
                rank_similarity[number2] = 0
        rank = np.array(rank_similarity)
        rank_sort = np.argsort(rank)
        rank_sort = rank_sort[-1::-1]

        rank2 = np.array(distance_similarity)
        rank_sort2 = np.argsort(rank2)
        rank_sort2 = rank_sort2[-1::-1]

        rank3 = np.array(distance_similarity2)
        rank_sort3 = np.argsort(rank3)
        rank_sort3 = rank_sort3[-1::-1]

        rank4 = np.array(distance_similarity3)
        rank_sort4 = np.argsort(rank4)
        rank_sort4 = rank_sort4[-1::-1]
        if len(answer[i]) >= 1:
            feature[i].append(rank[rank_sort[0]])
            feature[i].append(rank2[rank_sort2[0]])
            feature[i].append(rank3[rank_sort3[0]])
            feature[i].append(rank4[rank_sort4[0]])
        else:
            for number3 in range(3):
                feature[i].append(0)
                feature[i].append(0)
                feature[i].append(0)
                feature[i].append(0)
            continue
        if len(answer[i]) >= 2:
            feature[i].append(rank[rank_sort[1]])
            feature[i].append(rank2[rank_sort2[1]])
            feature[i].append(rank3[rank_sort3[1]])
            feature[i].append(rank4[rank_sort4[1]])
        else:
            for number3 in range(2):
                feature[i].append(0)
                feature[i].append(0)
                feature[i].append(0)
                feature[i].append(0)
            continue
        if len(answer[i]) >= 3:
            feature[i].append(rank[rank_sort[2]])
            feature[i].append(rank2[rank_sort2[2]])
            feature[i].append(rank3[rank_sort3[2]])
            feature[i].append(rank4[rank_sort4[2]])
        else:
            for number3 in range(1):
                feature[i].append(0)
                feature[i].append(0)
                feature[i].append(0)
                feature[i].append(0)
            continue

    return feature


def metadata_feature(question, answer):
    # 这里度量的是问题和答案的长度，以及他们的相对长度，以此作为其feature
    question_data = []
    answer_data = []
    for i in range(len(question)):
        for j in range(len(answer[i])):
            question_data.append(question[i])
            answer_data.append(answer[i][j])

    question_len = []
    answer_len = []
    for i in range(len(question_data)):
        question_len.append(len(question_data[i]))

    for i in range(len(answer_data)):
        answer_len.append(len(answer_data[i]))

    feature = []
    for i in range(len(question_data)):
        feature.append([])
        feature[i].append(answer_len[i])
        feature[i].append(question_len[i])
        if answer_len[i] != 0:
            feature[i].append(question_len[i] / answer_len[i])
        else:
            feature[i].append(0)
        if question_len[i] != 0:
            feature[i].append(answer_len[i] / question_len[i])
        else:
            feature[i].append(0)

    return feature


def frequency_statistical(train_file_dir, train_file_name):
    # 统计词频
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    lines = f.readlines()
    word = {}
    count = 0
    for line in lines:
        a = line.replace("\n", "").split("	")
        if len(a) == 1:
            continue
        word[a[0]] = a[1]
        count = count + 1
    return word


def get_word_frequency(start, end, q, item, word):
    # 将输入的单词可以转换为词频而不是word2vec
    question_matrix = []
    answer_matrix = []

    for i in range(start, end, 1):
        question_matrix.append([])
        for j in range(len(q[i])):
            if q[i][j] in word.keys():
                question_matrix[i - start].append(word[str(q[i][j])])

    for i in range(start, end, 1):
        answer_matrix.append([])
        for j in range(len(item[i])):
            answer_matrix[i - start].append([])
            for k in range(len(item[i][j])):
                if item[i][j][k] in word.keys():
                    answer_matrix[i - start][j].append(word[str(item[i][j][k])])
    return question_matrix, answer_matrix


def frequency_feature(start, end, q, item, word):
    # 在这里我们度量的是问题和答案中的单词的词频特征
    # 一般具有较低的词频越多的问题，就越好，因为越相近
    question, answer = get_word_frequency(start, end, q, item, word)
    question_data = []
    answer_data = []
    for i in range(len(question)):
        for j in range(len(answer[i])):
            question_data.append(question[i])

    for i in range(len(answer)):
        for j in range(len(answer[i])):
            answer_data.append(answer[i][j])

    feature = []
    for i in range(len(question_data)):
        feature.append([])
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for j in range(len(question_data[i])):
            if question_data[i][j] < 100:
                count1 += 1
            if question_data[i][j] >= 100 and question_data[i][j] < 500:
                count2 += 1
            if question_data[i][j] >= 500 and question_data[i][j] < 1000:
                count3 += 1
            if question_data[i][j] >= 1000:
                count4 += 1

        for k in range(len(answer_data[i])):
            if answer_data[i][k] < 100:
                count1 += 1
            if answer_data[i][k] >= 100 and answer_data[i][k] < 500:
                count2 += 1
            if answer_data[i][k] >= 500 and answer_data[i][k] < 1000:
                count3 += 1
            if answer_data[i][k] >= 1000:
                count4 += 1
        feature[i].append(count1)
        feature[i].append(count2)
        feature[i].append(count3)
        feature[i].append(count4)
    return feature


def same_word_feature(question_data, answer_data):
    # 这个feature度量的是在答案中有多少词是和问题是一样的
    # 我们认为问题和答案包含的相同的词越多，那么就越相似
    # 而且在这里衡量相同的词与不同的词的个数的比值
    feature = []
    for i in range(len(question_data)):
        feature.append([])
        count_same = 0
        count_diff = 0
        # question_data[i]是一个问题
        for j in range(len(question_data[i])):
            temp_vector = question_data[i][j]
            flag = 1
            for k in range(len(answer_data[i])):
                for mm in range(128):
                    if answer_data[i][k][mm] != temp_vector[mm]:
                        flag = 0
                if flag == 1:
                    count_same += 1
                    continue
                else:
                    count_diff += 1
                flag = 1
        feature[i].append(count_same)
        feature[i].append(count_diff)
        if count_diff != 0:
            feature[i].append(count_same * 1.0 / count_diff)
        else:
            feature[i].append(0)

        if count_same != 0:
            feature[i].append(count_diff * 1.0 / count_same)
        else:
            feature[i].append(0)
    return feature


if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'train.2.json'
    f3 = file('q.pkl', 'rb')
    f4 = file('item.pkl', 'rb')
    q = pickle.load(f3)
    item = pickle.load(f4)
    f3.close()
    f4.close()
    train_file_name = 'CCIR_train_2_word_num.txt'
    word = frequency_statistical(train_file_dir, train_file_name)
    start = 18
    end = 20
    feature = frequency_feature(start, end, q, item, word)
    print feature




