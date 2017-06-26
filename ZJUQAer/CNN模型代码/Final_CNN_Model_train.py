# -*-coding: utf-8 -*-
from keras.layers import Input, Dense, Dropout, Flatten, merge,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import hinge
from keras.models import load_model, model_from_json
from keras import regularizers
from keras import initializers
from keras.utils.np_utils import to_categorical
from evaluate import *
from frequency import *
import pickle
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 本程序是使用CNN来训练模型，其中对于词频大于50的词的忽略，并且将模型存储为'my_model_architecture.json','my_model_weights.h5'
# CNN模型通过两路输入，分别是答案和问题的，经过word2vec的矩阵表示。最后两路输出两个vector，进行点积得出分数


def cnn(height_a, height_q, width_a, width_q):
    question_input = Input(shape=(height_q, width_q, 1), name='question_input')
    conv1_Q = Conv2D(512, (2, 128), activation='sigmoid', padding='valid',
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01))(question_input)
    Max1_Q = MaxPooling2D((29, 1), strides=(1, 1), padding='valid')(conv1_Q)
    F1_Q = Flatten()(Max1_Q)
    Drop1_Q = Dropout(0.5)(F1_Q)
    predictQ = Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01))(Drop1_Q)


    # kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01)
    answer_input = Input(shape=(height_a, width_a, 1), name='answer_input')
    conv1_A = Conv2D(512, (2, 128), activation='sigmoid', padding='valid',
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01))(answer_input)
    Max1_A = MaxPooling2D((319, 1), strides=(1, 1), padding='valid')(conv1_A)
    F1_A = Flatten()(Max1_A)
    Drop1_A = Dropout(0.5)(F1_A)
    predictA = Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.01))(Drop1_A)

    predictions = merge([predictA, predictQ], mode='dot')
    model = Model(inputs=[question_input, answer_input],
                  outputs=predictions)

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    # model.compile(loss='mean_squared_error',
    #             optimizer='nadam')
    return model

if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    # f1 = file('q.pkl', 'wb')
    # f2 = file('item.pkl', 'wb')
    # pickle.dump(q, f1, True)
    # pickle.dump(item, f2, True)
    # f1.close()
    # f2.close()
    # f3 = file('q.pkl', 'rb')
    # f4 = file('item.pkl', 'rb')
    # q = pickle.load(f3)
    # item = pickle.load(f4)
    # f3.close()
    # f4.close()

    train_file_name = 'CCIR_test_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)

    length_a = 320
    length_q = 30
    print length_a, length_q
    model = cnn(length_a, length_q, 128, 128)
    # model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')

    for echo in range(7998):
        print 'the echo is', echo
        data_question, data_answer, height_a, height_q, width_a, width_q = get_train_data(echo*5, length_a,length_q, q, item, word)
        label = get_label(echo*5)
        print len(data_answer[0])
        if len(data_answer[0]) == 320:
            # model.fit([data_question, data_answer], label, batch_size=5, nb_epoch=2)
            t = model.train_on_batch([data_question, data_answer], label)
            print 'loss=', t
            json_string = model.to_json()
            open('my_model_architecture.json', 'w').write(json_string)
            model.save_weights('my_model_weights.h5')
        del data_answer
        del data_question
        del label
    model.save_weights('my_model_weights.h5')
    del q
    del item
    print(model.summary())

