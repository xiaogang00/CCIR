1、由维基百科的数据得到相关的分词文件
2、执行get_words_for_test_json.py来获取训练集及测试集的分词结果添加到维基百科分词文件的后面
3、执行train_word2vec_model.py结合维基百科的分词与测试集的分词得到word2vec模型
4、得CNN的模型的部分:
   1）训练：执行Final_CNN_Model_train.py进行cnn模型的训练，结果保存为my_model_weights.h5
      和my_model_architecture.json
   2）测试：执行Final_CNN_Model_test.py对测试集得到每一个问题对应的不同答案的分数
5、执行wordoverlap.py得到wordoverlap对于测试集的分词结果
6、执行get_CNN_wordoverlap_score.py获得wordoverlap的分数结果
7、执行get_latest_result.py获取最终的result.json

从训练数据的label统计情况可以看出，问题的答案的label分布主要是0和2比较多，同时数目较为接近，所以对于跑出的每个问题的所有答案的score分布，若score值过于分散或过于集中，则使用random的方法来重新获取该问题的score值