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

# 本程序是使用CNN训练h好的模型来进行测试，
# 调用的模型为'my_model_weights.h5'。

def compute_score(model, length_a, length_q):
    train_file_dir = './'
    train_file_name = 'json.testset'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    f1 = file('q.pkl', 'wb')
    f2 = file('item.pkl', 'wb')
    pickle.dump(q, f1, True)
    pickle.dump(item, f2, True)
    f1.close()
    f2.close()
    qaid = process_id(train_file_dir, train_file_name)
    print len(qaid)
    print len(q)
    print len(item)
    train_file_name = 'CCIR_test_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)
    
    for echo in range(1998):
        print 'the echo is', echo
        if echo != 100000:
            [test_question, test_answer, queryid] = get_test_data(length_a, length_q, echo*5, q, item, qaid, word)
        else:
            test_question = np.zeros((5, 1))
        for i in range(len(test_question)):
            if echo != 100000:
                temp = model.predict([test_question[i], test_answer[i]], batch_size=len(test_question[i]))
            else:
                temp = []
                for temp_number in range(len(test_question[i])):
                    temp.append(np.random.uniform(0, 2))
            if echo != 100000:
                temp_score = []
                for my_number2 in range(len(temp)):
                    temp_score.append(temp[my_number2][0])
                temp_score = np.array(temp_score)
            else:
                temp_score = np.array(temp)
            temp_id = queryid[i]
            if not os.path.exists('./Final'):
                os.makedirs('./Final')
            file_object = open('./Final/%d' % int(temp_id), 'w')
            print temp_score
            for my_number in range(len(temp_score)):
                a = "%lf\n" % (temp_score[my_number])
                file_object.write(a)
    del q
    del item
    M = temp_score
    return M


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
    return model

if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'

    length_a = 320
    length_q = 30
    print length_a, length_q
    model = cnn(length_a, length_q, 128, 128)
    model.load_weights('my_model_weights.h5')
    M = compute_score(model, length_a, length_q)
    print M

