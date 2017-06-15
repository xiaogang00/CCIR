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
from evaluate3 import *
from frequency import *
from feature2 import *
import pickle
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#K.set_session(session)x

def compute_score(model, length_a, length_q):
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'
    #[q, item] = process_train_file(train_file_dir, train_file_name)
    f3 = file('q.pkl', 'rb')
    f4 = file('item.pkl', 'rb')
    q = pickle.load(f3)
    item = pickle.load(f4)
    f3.close()
    f4.close()
    train_file_name = 'CCIR_train_3_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)
    #f = open(os.path.join(train_file_dir, train_file_name), 'r')
    Total_score_dcg3 = []
    Total_score_dcg5 = []
    Best_score_dcg3 = []
    Best_score_dcg5 = []
    #for i in range(5551):
    #    Total_score_dcg3.append(4.00770684819)
    #    Total_score_dcg5.append(5.46699756308)
    for echo in range(7998):
        count = 0
        print 'the echo is', echo
        if echo != 100000:
            [test_question, test_answer] = get_test_data(length_a, length_q, echo * 5, q, item, word)
            feature = get_feature_data(echo * 5, echo * 5 + 5, q, item, word)
        else:
            test_question = np.zeros((5, 1))
        test_label = get_test_label(echo*5)
        for i in range(len(test_question)):
            if echo != 100000:
                length_feature = len(test_question[i])
                temp_feature = []
                for nnn in range(length_feature):
                    temp_feature.append(feature[count])
                    count += 1
                temp_feature = np.array(temp_feature)
                temp = model.predict([test_question[i], test_answer[i], temp_feature], batch_size=len(test_question[i]))
            else:
                temp = test_label[i]
            #print len(test_question[i]),len(test_answer[i])
            #print temp
            temp_label = test_label[i]
            if echo != 100000:
                temp_score = []
                for my_number2 in range(len(temp)):
                    temp_score.append(temp[my_number2][0])
                temp_score = np.array(temp_score)
            else:
                temp_score = np.array(temp)
            if not os.path.exists('./CNNModel4'):
                os.makedirs('./CNNModel4')
            file_object = open('./CNNModel4/%d' % (echo * 5 + i), 'w')
            print temp_score
            for my_number in range(len(temp_score)):
                a = "%d  %lf \n" % (temp_label[my_number], temp_score[my_number])
                file_object.write(a)

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

            best_label = np.array(temp_label)
            temp_sort2 = np.argsort(best_label)
            temp_sort2 = temp_sort2[-1::-1]
            Best_Dcg3 = 0
            Best_Dcg5 = 0
            for number in range(1, 4, 1):
                a = temp_sort2[number-1]
                a = int(a)
                Best_Dcg3 = Best_Dcg3 + (np.power(2, temp_label[a])-1) / np.log2(number+1)
            for number in range(1, 6, 1):
                a = temp_sort2[number-1]
                a = int(a)
                Best_Dcg5 = Best_Dcg5 + (np.power(2, temp_label[a])-1) / np.log2(number+1)
            Best_score_dcg3.append(Best_Dcg3)
            Best_score_dcg5.append(Best_Dcg5)
        print 'The score for Dcg3 is', np.mean(Total_score_dcg3)
        print 'The score for Dcg5 is', np.mean(Total_score_dcg5)
        print 'The best score for Dcg3 is', np.mean(Best_score_dcg3)
        print 'The best score for Dcg5 is', np.mean(Best_score_dcg5)
    del q
    del item
    M = np.mean(Total_score_dcg3)
    return M


def Margin_Loss(y_true, y_pred):
    score_best = y_pred[0]
    score_predict = y_pred[1]
    loss = K.maximum(0.0, 1.0 - K.sigmoid(score_best - score_predict))
    return K.mean(loss) + 0 * y_true


def cnn(height_a, height_q, width_a, width_q, extra_len):
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

    extra_input = Input(shape=(extra_len,), name='extra_input')
    predictQ1 = concatenate([predictQ, extra_input], axis=1)
    predictA1 = concatenate([predictA, extra_input], axis=1)
    predictions = merge([predictA1, predictQ1], mode='dot')
    model = Model(inputs=[question_input, answer_input, extra_input],
                  outputs=predictions)

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    # model.compile(loss='mean_squared_error',
    #             optimizer='nadam')
    return model

if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'
    # [q, item] = process_train_file(train_file_dir, train_file_name)
    # f1 = file('q.pkl', 'wb')
    # f2 = file('item.pkl', 'wb')
    # pickle.dump(q, f1, True)
    # pickle.dump(item, f2, True)
    # f1.close()
    # f2.close()
    f3 = file('q.pkl', 'rb')
    f4 = file('item.pkl', 'rb')
    q = pickle.load(f3)
    item = pickle.load(f4)
    f3.close()
    f4.close()

    train_file_name = 'CCIR_train_3_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)
    # f = open(os.path.join(train_file_dir, train_file_name), 'r')
    # length_a, length_q = find_max(q, item, word)
    # print length_a, length_q

    length_a = 320
    length_q = 30
    print length_a, length_q
    model = cnn(length_a, length_q, 128, 128, 24)
    # model = model_from_json(open('my_model_architecture4.json').read())
    # model.load_weights('my_model_weights4.h5')
    # 1 - 39997
    for echo in range(7898):
        print 'the echo is', echo
        data_question, data_answer, height_a, height_q, width_a, width_q = get_train_data(echo*5, length_a, length_q, q, item, word)
        label = get_label(echo*5)
        feature = get_feature_data(echo*5, echo*5+5, q, item, word)
        feature = np.array(feature)
        print len(data_answer[0])
        if len(data_answer[0]) == 320:
            # model.fit([data_question, data_answer], label, batch_size=5, nb_epoch=2)
            t = model.train_on_batch([data_question, data_answer, feature], label)
            print 'loss=', t
            json_string = model.to_json()
            open('my_model_architecture4.json', 'w').write(json_string)
            model.save_weights('my_model_weights4.h5')
        del data_answer
        del data_question
        del label
    # model.save('my_model3.h5')
    # model.save_weights('my_model_weights4.h5')
    del q
    del item
    # print(model.summary())
    M = compute_score(model, length_a, length_q)
    print M
