#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于读取训练数据，对于每一组训练数据中的问题和答案，都分好词之后，放入一个sentences的list中，然后将该list中的句子进行分词，分词之后放入一个word的list中
#对于words的list，用corpora.Dictionary变成一个dict，然后生成语料库，及tfidf模型，之后生成LDA模型及corpus_lda
#再输入每个问题的query语句进行分词之后，用lda,sims确认与哪个答案的关系最近，把那个答案的label放在前面

#sys.argv[1]为wordoverlap所分解出的问题与答案的单词文件的目录，sys.argv[2]为训练数据中共有的问题数目
#sys.argv[3]为所要求的score值存储的文件目录

#本程序专门用来获取不用排序下的各个问题的答案按照答案的顺序下的score与label值

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from gensim import corpora,models,similarities

import os
import numpy as np

def calu_DCG(answer_label_list,k):
    k_temp = 0
    DCG_result = 0
    for answer_label in answer_label_list:
        if k_temp < int(k): 
            label = int(answer_label)
            log_num = np.log(k_temp+2)/np.log(2)
            DCG_result = DCG_result + (2**label - 1)/log_num
            k_temp = k_temp + 1
        else:
            break
    return DCG_result

def calu_avg_answer_length(answer_length):
    length_sum = 0
    for length in answer_length:
        length_sum = length_sum + length
    avg_length = length_sum/len(answer_length)
    return avg_length
    
def get_score_for_question(question_answer_word_dir,question_num,question_answer_score_label_file_dir):
    DCG_score_list = []
    for question_index in range(int(question_num)):
        if (question_index+1)%1000 == 1:
            print 'Now for line : ' + str(question_index+1) + '\n'
        index = question_index + 1
        file_read_name = os.path.join(question_answer_word_dir,str(index))
        file_write_name = os.path.join(question_answer_score_label_file_dir,str(index))
        file_read = open(file_read_name,'rb+')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        question_line_list.remove('question')
        answer_index = 0
        answer_index_line_label_dict = {}
        answer_sentences_word_list = []
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = answer_temp_line_list[1]
            answer_temp_line_list.remove('answer')
            answer_temp_line_list.remove(answer_label)
            answer_sentences_word_list.append(answer_temp_line_list)
            answer_list_temp = []
            answer_list_temp.append(answer_label)
            answer_index_line_label_dict[answer_index] = answer_list_temp
            answer_index += 1
        dic = corpora.Dictionary(answer_sentences_word_list)
        corpus=[dic.doc2bow(text) for text in answer_sentences_word_list]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lda = models.LdaModel(corpus_tfidf,id2word=dic,num_topics=2)
        index = similarities.MatrixSimilarity(lda[corpus_tfidf])
        query_bow = dic.doc2bow(question_line_list)
        query_lda = lda[query_bow]
        sims = index[query_lda]
        list_simes = list(enumerate(sims))
        sort_sims = sorted(enumerate(sims),key=lambda item:-item[1])
        #answer_label_list = []
        for item in list_simes:
            answer_index_temp = item[0]
            answer_label = int(answer_index_line_label_dict[int(answer_index_temp)][0])
            answer_score = str(item[1])
            file_write = open(file_write_name,'ab+')
            file_write.write(str(answer_label)+'\t'+str(answer_score)+'\n')
            file_write.close()
            #answer_label_list.append(answer_label)
        #DCG_score = calu_DCG(answer_label_list,k)
        #DCG_score_list.append(DCG_score)
    #DCG_avg = calu_avg_answer_length(DCG_score_list)
    #print 'DCG_avg : \t' + str(DCG_avg)
        

def main():
    question_answer_word_dir = sys.argv[1]
    question_num = sys.argv[2]
    #k = sys.argv[3]
    question_answer_score_label_file_dir = sys.argv[3]
    get_score_for_question(question_answer_word_dir,int(question_num),question_answer_score_label_file_dir)

if __name__ == '__main__':
    main()