def modelcnn(dim, max_ques_len, max_ans_len, vocab_lim, embedding):
    inp_q = Input(shape=(max_ques_len,))
    embedding_q = Embedding(input_dim=vocab_lim, output_dim=dim, input_length=max_ques_len, weights=[embedding], trainable=False)(inp_q)
    conv_q = Convolution1D(500, 20, border_mode='same', activation='sigmoid')(embedding_q)
    conv_q = Dropout(0.25)(conv_q)
    #conv_q = Convolution1D(500, 20, border_mode='same', activation='sigmoid')(conv_q)
    conv_q= Dense(500)(conv_q)
    pool_q = GlobalMaxPooling1D()(conv_q)

    #pool_q=Flatten()(conv_q)
    #pool_q=Dense(1000)(pool_q)

    inp_a = Input(shape=(max_ans_len,))
    embedding_a = Embedding(vocab_lim, dim, input_length=max_ans_len, weights=[embedding], trainable=False)(inp_a)
    conv_a = Convolution1D(500, 20, border_mode='same', activation='sigmoid')(embedding_a)
    conv_a = Dropout(0.25)(conv_a)
    #conv_a = Convolution1D(500, 20, border_mode='same', activation='sigmoid')(conv_a)
    conv_a = Dense(500)(conv_a)
    pool_a = GlobalMaxPooling1D()(conv_a)

    #pool_a=Flatten()(conv_a)
    #pool_a = Dense(1000)(pool_a)

    # sim = merge([Dense(500, bias=False)(pool_q), pool_a], mode='dot')
    # # print pool_a, pool_q
    #
    # model_sim = merge([pool_q, pool_a, sim], mode='concat')
    # print (model_sim)

    #model_final = Flatten()(model_sim)

    #model_final = Dropout(0.2)(model_sim)
    #model_final = Dense(701)(model_sim)
    #model_final = Dense(200)(model_final)
    #model_final = Dropout(0.2)(model_final)
    #model_final = Dense(1, activation='sigmoid')(model_final)


    #model_final=Dense(1, activation='sigmoid')(merge([Dense(500, bias=False)(pool_q), pool_a], mode='dot'))

    dot = lambda a, b: K.batch_dot(a, b, axes=1)
    calc_sim=lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())

    #pos_similarity=keras.layers.merge([pool_q, pool_ans_pos], mode=calc_sim,output_shape=lambda x: x[0])
    similarity=keras.layers.merge([pool_q, pool_a], mode=calc_sim,output_shape=lambda x: (None, 1))

    model = Model(input=[inp_q, inp_a], output=[similarity])
    #model = Model(input=[inp_q, inp_a], output=[model_final])
    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    print(model.output_shape)
    #model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.compile(loss='squared_hinge', optimizer='nadam', metrics=['accuracy'])
    #model.compile(loss='hinge', optimizer='nadam', metrics=['accuracy'])
    #model.compile(loss=lambda y_true, y_pred: K.mean(K.maximum(1. - 1.1*2.0*(y_true-0.5) * 2.0*(y_pred-0.5), 0.), axis=-1), optimizer='nadam', metrics=['acc'])
    print (model.summary())
    return model

model=modelcnn(300, 383, 383, 52029, embedding_matrix)
from qacnn import  cnn2
#model=cnn2(383,383,52029,embedding_matrix)
#cnn2(max_ques_len, max_ans_len,wordnum,weights):
model.fit([ques_matrix, ans_matrix], [label], batch_size=32, epochs=4)