# -*-coding: utf-8 -*-
from keras.layers import Input, Dense, Dropout, Flatten, merge,concatenate
from keras.layers import Conv2D, MaxPooling2D, Embedding, Reshape
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import hinge
from keras.models import load_model, model_from_json
from keras import regularizers
from keras import initializers
from keras.utils.np_utils import to_categorical
from get_words_for_CCIR import *
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def compute_score(model, length_a, length_q, q, item):
    train_file_dir = './'
    train_file_name = 'train.1.json'
    # [q, item] = process_train_file(train_file_dir, train_file_name)
    train_file_name = 'CCIR_train_word_num.txt'
    word, word_remove, count = word_table(train_file_dir, train_file_name)
    Total_score_dcg3 = []
    Total_score_dcg5 = []
    for echo in range(100):
        print 'the echo is', echo
        [test_question, test_answer] = get_test_data(length_a, length_q, echo*5+4000, q, item, word)
        test_label = get_test_label(echo*5+4000)
        for i in range(len(test_question)):
            temp = model.predict([test_question[i], test_answer[i]], batch_size=len(test_question[i]))
            print len(test_question[i]), len(test_answer[i])
            temp_label = test_label[i]
            temp_score = []
            for my_number2 in range(len(temp)):
                temp_score.append(temp[my_number2][0])
            temp_score = np.array(temp_score)
            # 在这里将我们测试出来的score和最后的label写入文件
            #if not os.path.exists('./CNNModel2'):
            #    os.makedirs('./CNNModel2')
            #file_object = open('./CNNModel2/%d' % (echo*5+i), 'w')
            print temp_score
            #for my_number in range(len(temp_score)):
            #    a = "%d  %lf \n" % (temp_label[my_number], temp_score[my_number])
            #    file_object.write(a)
            temp_sort = np.argsort(temp_score)
            temp_sort = temp_sort[-1::-1]
            Dcg3 = 0
            Dcg5 = 0
            print temp_label
            for number in range(1, 4, 1):
                a = temp_sort[number-1]
                a = int(a)
                Dcg3 = Dcg3 + (np.power(2, temp_label[a])-1) / np.log2(number+1)
            for number in range(1, 6, 1):
                a = temp_sort[number-1]
                a = int(a)
                Dcg5 = Dcg5 + (np.power(2, temp_label[a])-1) / np.log2(number+1)
            Total_score_dcg3.append(Dcg3)
            Total_score_dcg5.append(Dcg5)
        print 'The score for Dcg3 is', np.mean(Total_score_dcg3)
        print 'The score for Dcg5 is', np.mean(Total_score_dcg5)
    del q
    del item
    M = np.mean(Total_score_dcg3)
    print M
    return M


def get_test_data(length_a, length_q, start, q, item, word):
    question, answer = get_word_vector(start, q, item, word, word_remove, start + 5)
    answer2, question2 = my_padding(question, answer, length_a, length_q)
    question_new = []
    for i in range(len(question2)):
        question_new.append([])
        for j in range(len(answer2[i])):
            question_new[i].append(question2[i])
    del question
    del question2
    height_q = len(question_new[0][0])
    height_a = len(answer2[0][0])

    data_question = np.array(question_new)
    data_answer = np.array(answer2)
    question_data = []
    for i in range(len(question_new)):
        data_question[i] = np.array(data_question[i])
        temp = data_question[i].reshape((len(question_new[i]), height_q, 1))
        question_data.append(temp)

    answer_data = []
    for i in range(len(answer2)):
        data_answer[i] = np.array(data_answer[i])
        temp = data_answer[i].reshape((len(answer2[i]), height_a, 1))
        answer_data.append(temp)
    return question_data, answer_data


def word_table(train_file_dir,train_file_name):
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    lines = f.readlines()
    word = {}
    word_remove = []
    count = 0
    for line in lines:
        a = line.replace("\n","").split("	")
        if len(a) == 1:
            continue
        if int(a[1]) < 100:
            count = count + 1
            word[str(a[0])] = int(count)
            continue
        word_remove.append(a[0])
    #for i in range(len(word)):
    #    print word[i]
    return word, word_remove, count


def my_padding(data_q, data_a, length_a, length_q):
    if length_q != 0 and length_a != 0:
        for i in range(len(data_q)):
            m = len(data_q[i])
            a = int((length_q - m)/2)
            b = length_q - m - a
            for j in range(a):
                temp = 0
                data_q[i].append(temp)
            for j in range(b):
                temp = 0
                data_q[i].insert(0, temp)

        for i in range(len(data_a)):
            for j in range(len(data_a[i])):
                m = len(data_a[i][j])
                a = int((length_a-m))/2
                b = length_a-m-a
                for number in range(a):
                    temp = 0
                    data_a[i][j].append(temp)
                for number in range(b):
                    temp = 0
                    data_a[i][j].insert(0, temp)
    return data_a, data_q


def get_word_vector(start, q, item, word, word_remove, end):
    question_matrix = []
    answer_matrix = []
    for i in range(start, end, 1):
        question_matrix.append([])
        for j in range(len(q[i])):
            if q[i][j] in word_remove:
                continue
            if q[i][j] in word.keys():
                question_matrix[i - start].append(word[str(q[i][j])])

    for i in range(start, end, 1):
        answer_matrix.append([])
        for j in range(len(item[i])):
            answer_matrix[i - start].append([])
            for k in range(len(item[i][j])):
                if item[i][j][k] in word_remove:
                    continue
                if item[i][j][k] in word.keys():
                    answer_matrix[i - start][j].append(word[str(item[i][j][k])])

    return question_matrix, answer_matrix


def get_train_data(start, lengtha, lengthq, q, item, word, word_remove):
    question, answer = get_word_vector(start, q, item, word, word_remove, start+1)
    answer2, question2 = my_padding(question, answer, lengtha, lengthq)
    question_new = []
    for i in range(len(question2)):
        question_new.append([])
        for j in range(len(answer2[i])):
            question_new[i].append(question2[i])

    del question
    del answer
    del question2
    final_question = []
    final_answer = []
    height_q = len(question_new[0][0])
    height_a = len(answer2[0][0])
    for i in range(len(question_new)):
        for j in range(len(question_new[i])):
            temp1 = np.array(question_new[i][j])
            temp2 = temp1.reshape(height_q, 1)
            final_question.append(temp2)

    for i in range(len(answer2)):
        for j in range(len(answer2[i])):
            temp1 = np.array(answer2[i][j])
            temp2 = temp1.reshape(height_a, 1)
            final_answer.append(temp2)
    del answer2
    del question_new
    final_answer = np.array(final_answer)
    final_question = np.array(final_question)
    return final_question, final_answer, height_a, height_q


def cnn(height_a, height_q, count):
    question_input = Input(shape=(height_q, 1), name='question_input')
    embedding_q = Embedding(input_dim=count, output_dim=128,  input_length=height_q)(question_input)
    re_q = Reshape((height_q, 128, 1), input_shape=(height_q,))(embedding_q)
    conv1_Q = Conv2D(512, (2, 128), activation='sigmoid', padding='valid',
                     kernel_regularizer=regularizers.l2(0.02),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05))(re_q)
    Max1_Q = MaxPooling2D((29, 1), strides=(1, 1), padding='valid')(conv1_Q)
    F1_Q = Flatten()(Max1_Q)
    Drop1_Q = Dropout(0.5)(F1_Q)
    predictQ = Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05))(Drop1_Q)


    # kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01)
    answer_input = Input(shape=(height_a, 1), name='answer_input')
    embedding_a = Embedding(input_dim=count, output_dim=128,  input_length=height_a)(answer_input)
    re_a = Reshape((height_a, 128, 1), input_shape=(height_a,))(embedding_a)
    conv1_A = Conv2D(512, (2, 128), activation='sigmoid', padding='valid',
                     kernel_regularizer=regularizers.l2(0.02),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05))(re_a)
    Max1_A = MaxPooling2D((399, 1), strides=(1, 1), padding='valid')(conv1_A)
    F1_A = Flatten()(Max1_A)
    Drop1_A = Dropout(0.5)(F1_A)
    predictA = Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.02),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05))(Drop1_A)

    predictions = merge([predictA, predictQ], mode='dot')
    model = Model(inputs=[question_input, answer_input],
                  outputs=predictions)

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    # model.compile(loss='mean_squared_error',
    #             optimizer='nadam')
    return model


def get_label(start):
    train_file_dir = './'
    train_file_name = 'train.1.json'
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    label = []
    count = 0
    for line in f:
        if count < start:
            count = count + 1
            continue
        if count >= start + 1:
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
            label[count-start].append(temp)
        count = count + 1
    final_label = []
    for i in range(len(label)):
        for j in range(len(label[i])):
            final_label.append(label[i][j])
    del label
    final_label = np.array(final_label)
    return final_label

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
        if count >= start + 5:
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

if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'train.1.json'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    train_file_name = 'CCIR_train_word_num.txt'
    word, word_remove, count = word_table(train_file_dir, train_file_name)
    # a = word.keys()
    height_a = 400
    height_q = 30
    #model = cnn(height_a, height_q, count)
    model = model_from_json(open('my_model_architecture3.json').read())
    model.load_weights('my_model_weights3.h5')
    for echo in range(4000):
        continue
        if echo == 2392:
            continue
        print 'the echo is', echo
        final_question, final_answer, height_a, height_q = get_train_data(echo, height_a, height_q, q, item,
                                                                          word, word_remove)
        label = get_label(echo)
        t = model.train_on_batch([final_question, final_answer], label)
        print 'loss=', t
        json_string = model.to_json()
        open('my_model_architecture3.json', 'w').write(json_string)
        model.save_weights('my_model_weights3.h5')
        del final_question
        del final_answer
        del label
    #model.save('my_model3.h5')
    #model.save_weights('my_model_weights3.h5')
    #print(model.summary())
    M = compute_score(model, height_a, height_q, q, item)