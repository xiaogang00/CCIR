#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于计算对于train1.json的最佳的score值

#sys.argv[1]为wordoverlap所分解出的问题与答案的单词文件的目录，sys.argv[2]为训练数据中共有的问题数目
#sys.argv[3]为所要求得DCG@的值

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

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
    
def get_score_for_question(question_answer_word_dir,question_num,k):
    DCG_score_list = []
    for question_index in range(int(question_num)):
        if (question_index+1)%1000 == 1:
            print 'Now for line : ' + str(question_index+1) + '\n'
        index = question_index + 1
        file_read_name = os.path.join(question_answer_word_dir,str(index))
        file_read = open(file_read_name,'rb+')
        question_line = file_read.readline()
        label_list = []
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = int(answer_temp_line_list[1])
            label_list.append(answer_label)
        label_list.sort(reverse=True)
        DCG_score = calu_DCG(label_list,k)
        DCG_score_list.append(DCG_score)
    DCG_avg = calu_avg_answer_length(DCG_score_list)
    print 'DCG_avg : \t' + str(DCG_avg)
    
def main():
    question_answer_word_dir = sys.argv[1]
    question_num = sys.argv[2]
    k = sys.argv[3]
    get_score_for_question(question_answer_word_dir,int(question_num),int(k))

if __name__ == '__main__':
    main()