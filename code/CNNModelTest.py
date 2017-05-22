from keras.layers import Input, Dense, Dropout, Flatten, merge,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import hinge
from keras.models import load_model, model_from_json
from evaluate2 import *
from frequency import *
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#K.set_session(session)

def compute_score(model, length_a, length_q):
    train_file_dir = './'
    train_file_name = 'train.1.json'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    train_file_name = 'CCIR_train_word_num.txt'
    word = frequency_word(train_file_dir, train_file_name)
    #f = open(os.path.join(train_file_dir, train_file_name), 'r')
    Total_score_dcg3 = []
    Total_score_dcg5 = []
    for echo in range(900):
        #if echo != 0:
        #    continue
        print 'the echo is', echo
        [test_question, test_answer] = get_test_data(length_a, length_q, echo*5, q, item, word)
        test_label = get_test_label(echo*5)
        for i in range(len(test_question)):
            #if i != 0:
            #    continue
            temp = model.predict([test_question[i], test_answer[i]], batch_size=len(test_question[i]))
            #layer_name = 'max_pooling2d_2'
            #layer_name = 'conv2d_4'
            #layer_name = 'flatten_1'
            #layer_name = 'dense_3'
            #intermediate_layer_model = Model(inputs=model.input,
            #                     outputs=model.get_layer(layer_name).output)
            #intermediate_output = intermediate_layer_model.predict([test_question[i], test_answer[i]])
            #intermediate_output = test_answer[i]
            #for layer in model.layers:
            #a = model.get_layer(layer_name)
            #weights = a.get_weights()
            #print weights

            #for number in range(len(intermediate_output)):
            #    print '-------------------------------------------------'
            #    for number2 in range(len(intermediate_output[number])):
            #        print intermediate_output[number][number2],

            #print len(intermediate_output)
            #for number in range(len(intermediate_output)):
            #    print '-------------------------------------------------'
            #    for number2 in range(len(intermediate_output[number])):
            #        if number2 != 0 and number2 != 1:
            #            continue
            #        count = 0
            #        for number3 in range(len(intermediate_output[number][number2])):
            #            if count % 10 == 0:
            #                print '\n'
            #            count = count + 1
            #            if intermediate_output[number][number2][number3] != 0:
            #                print intermediate_output[number][number2][number3],
            #        print '\n'
            #print '-----------------------'

            temp_label = test_label[i]
            temp_score = []
            for mynumber2 in range(len(temp)):
                temp_score.append(temp[mynumber2][0])
            temp_score = np.array(temp_score)
            if not os.path.exists('./CNNModel4'):
                os.makedirs('./CNNModel4')
            file_object = open('./CNNModel4/%d' % (echo * 5 + i), 'w')
            print temp_score
            print len(temp_score)
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
        print 'The score for Dcg3 is', np.mean(Total_score_dcg3)
        print 'The score for Dcg5 is', np.mean(Total_score_dcg5)
    del q
    del item
    M = np.mean(Total_score_dcg3)
    return M



def Margin_Loss(y_true, y_pred):
    score_best = y_pred[0]
    score_predict = y_pred[1]
    loss = K.maximum(0.0, 1.0 - K.sigmoid(score_best - score_predict))
    return K.mean(loss) + 0 * y_true


def cnn(height_a, height_q, width_a, width_q):
    question_input = Input(shape=(height_q, width_q, 1), name='question_input')
    conv1_Q = Conv2D(1, (16, 35), activation='relu', padding='valid')(question_input)
    conv2_Q = Conv2D(1, (10, 35), activation='relu', padding='valid')(conv1_Q)
    Max1_Q = MaxPooling2D((2, 35), strides=(1, 1), padding='valid')(conv2_Q)
    F1_Q = Flatten()(Max1_Q)
    X1_Q = Dense(64, activation='relu')(F1_Q)
    # Drop3_Q = Dropout(0.5)(X1_Q)
    predictQ = Dense(16, activation='relu')(X1_Q)

    answer_input = Input(shape=(height_a, width_a, 1), name='answer_input')
    conv1_A = Conv2D(1, (160, 35), activation='sigmoid', padding='valid',
                     kernel_regularizer=regularizers.l2(0.01))(answer_input)
    #Drop1_A = Dropout(0.25)(conv1_A)
    conv2_A = Conv2D(1, (80, 35), activation='sigmoid', padding='valid',
                     kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(conv1_A)
    Max1_A = MaxPooling2D((40, 35), strides=(1, 1), padding='valid')(conv2_A)
    F1_A = Flatten()(Max1_A)
    X1_A = Dense(256, activation='relu')(F1_A)
    Drop2_A = Dropout(0.5)(X1_A)
    X2_A = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l2(0.01))(Drop2_A)
    predictA = Dense(16, activation='relu')(X2_A)

    # merged_vector = concatenate([predictQ, predictA], axis=-1)
    # prediction_temp1 = Dense(10, activation='relu')(merged_vector)
    # prediction_temp2 = Dropout(0.25)(prediction_temp1)
    # predictions = Dense(1, activation='relu')(merged_vector)
    predictions = merge([predictA, predictQ], mode='dot')
    # predictions_best = merge([predictB, predictQ], mode='dot')

    model = Model(inputs=[question_input, answer_input],
                  outputs=predictions)
    #model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam())
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model

if __name__ == '__main__':
    # data_q = np.random.random((100, 100, 128, 1))
    # data_a = np.random.random((100, 100, 128, 1))
    # data_b = np.random.random((100, 100, 128, 1))
    # label = np.random.randint(2, size=(100, 1))
    # length_a, length_q = find_max()
    length_a = 320
    length_q = 30
    print length_a, length_q
    # model = cnn(length_a, length_q, 128, 128)
    model = model_from_json(open('my_model_architecture2.json').read())
    # model.load_weights('model_weight.h5')
    model.load_weights('my_model_weights2.h5')
    for echo in range(400):
        continue
        print 'the echo is', echo
        data_question, data_answer,height_a, height_q, width_a, width_q = get_train_data(echo, length_a,length_q)
        label = get_label(echo)
        model.train_on_batch([data_question, data_answer], label)
        #model.save('my_model.h5')
        model.save_weights('my_model_weights.h5')
        json_string = model.to_json() 
        open('my_model_architecture.json','w').write(json_string) 
        del data_answer
        del data_question
        del label
        # model.fit([data_question, data_answer], label, batch_size=5, nb_epoch=1)
    # model.save('my_model.h5')
    # model.save_weights('my_model_weights.h5')
    # print(model.summary())
    M = compute_score(model, length_a, length_q)
    print M

