#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于求wordembedding，将question及answer的各个单词的embedding求出，然后取问题和答案的向量的平均值作为该问题和答案的向量，然后求余弦距离，作为score值，存入对应问题的文件之中
#注意问题和答案之中的stopwords不求其向量，计算进入求解问题和答案的平均向量的过程之中
#问题和答案去除stopwords后的单词情况已经存储进入/home/dcd-qa/MLT_code/NLPCC/data/wordoverlap/question_ansers_words_without_stopwords/内的各个文件之中了

#sys.argv[1]为wordoverlap所分解出的问题与答案的单词文件的目录，sys.argv[2]为训练数据中共有的问题数目


import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import numpy as np
import re


def get_vector(word,model):
    vector = model[word]
    return vector
    
def get_mean_vector(question_line_list,answer_temp_line_list,model):
    #求问题与答案的平均向量
    question_vector_list = []
    answer_temp_vector_list = []
    question_vector = np.zeros(128)
    answer_vector = np.zeros(128)
    question_word_num = 0
    answer_word_num = 0
    for word in question_line_list:
        word = word.decode('utf-8')
        try:
            word_vector = get_vector(word,model)
            question_word_num = question_word_num + 1
            question_vector = question_vector + word_vector
        except KeyError:
            continue
    for word in answer_temp_line_list:
        word = word.decode('utf-8')
        try:
            word_vector = get_vector(word,model)
            answer_vector = answer_vector + word_vector
            answer_word_num = answer_word_num + 1
        except KeyError:
            continue
    question_mean_vector = (question_vector/question_word_num).tolist()
    answer_mean_vector = (answer_vector/answer_word_num).tolist()
    return question_mean_vector,answer_mean_vector
    
def calu_cosine_distance(question_vector,answer_temp_vector):
    dot = np.dot(question_vector,answer_temp_vector)
    question_length = np.sqrt(np.sum(np.square(question_vector)))
    answer_length = np.sqrt(np.sum(np.square(answer_temp_vector)))
    cosine_distance = dot/(float(question_length)*float(answer_length))
    return cosine_distance
    
def calu_score(question_line_list,answer_temp_line_list,model):
    question_vector,answer_temp_vector = get_mean_vector(question_line_list,answer_temp_line_list,model)
    score = calu_cosine_distance(question_vector,answer_temp_vector)
    return score
    
def get_score_for_question(question_answer_word_dir,question_num,model):
    for question_index in range(int(question_num)):
        if (question_index+1)%1000 == 1:
            print 'Now for line : ' + str(question_index+1) + '\n'
        index = question_index + 1
        file_read_name = os.path.join(question_answer_word_dir,str(index))
        file_read = open(file_read_name,'rb+')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        question_line_list.remove('question')
        file_write = open('/home/dcd-qa/MLT_code/NLPCC/data/word_embedding/answer_label_score/' + str(question_index+1),'wb')
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = answer_temp_line_list[1]
            answer_temp_line_list.remove('answer')
            answer_temp_line_list.remove(answer_label)
            answer_temp_score = calu_score(question_line_list,answer_temp_line_list,model)
            file_write = open('/home/dcd-qa/MLT_code/NLPCC/data/word_embedding/answer_label_score/' + str(question_index+1),'ab')
            file_write.write(answer_label + '\t' + str(answer_temp_score))
            file_write.write('\n')
            file_write.close()

def main():
    gensim_model = "/home/dcd-qa/MLT_code/NLPCC/data/word2vec/wiki.zh.text.model"
    model = gensim.models.Word2Vec.load(gensim_model)
    question_answer_word_dir = sys.argv[1]
    question_num = sys.argv[2]
    get_score_for_question(question_answer_word_dir,question_num,model)

if __name__ == '__main__':
    main()