#! /usr/env/bin python
# -*-coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np

def calu_avg_answer_length(answer_length):
    length_sum = 0
    for length in answer_length:
        length_sum = length_sum + length
    avg_length = length_sum/len(answer_length)
    return avg_length

def sort_by_value(d): 
    items=d.items() 
    backitems=[[v[1],v[0]] for v in items] 
    backitems.sort(reverse = True) 
    return [ backitems[i][1] for i in range(0,len(backitems))]
    
def calu_DCG(answer_score_file,k):
    #明天实现DCG的计算函数！！！！！！！！！
    answer_index_label_dict = {} #存储answer_index与label的对应dict
    answer_index_score_dict = {} #存储answer_index与score的对应dict
    #score_answer_index_dict = {} #存储score与answer_index的对应dict
    answer_index = 0
    for line in open(answer_score_file,'rb').readlines():
        line_list = line.strip().split('\t')
        label = int(line_list[0])
        score = float(line_list[1])
        answer_index_label_dict[answer_index] = label
        answer_index_score_dict[answer_index] = score
        #score_answer_index_dict[score] = []
        #score_answer_index_dict[score].append(answer_index) #用来防止出现多个answer有相同score值的情况
        answer_index = answer_index + 1
    answer_index_list = sort_by_value(answer_index_score_dict)
    k_temp = 0
    DCG_result = 0
    for answer_index in answer_index_list:
        if k_temp < int(k): 
            score = answer_index_score_dict[answer_index]
            label = answer_index_label_dict[answer_index]
            log_num = np.log(k_temp+2)/np.log(2)
            DCG_result = DCG_result + (2**label - 1)/log_num
            k_temp = k_temp + 1
    return DCG_result
    
def calu_avg_DCG(question_num,k):
    DCG_list = [] #存储每个question的DCG值
    for id in range(question_num):
        answer_score_file = '/home/dcd-qa/MLT_code/BM25/data/answer_score_label/' + str(id+1)
        DCG_temp = calu_DCG(answer_score_file,k)
        DCG_list.append(DCG_temp)
    DCG_avg = calu_avg_answer_length(DCG_list)
    return DCG_avg
        
def main():
    k = sys.argv[1]
    DCG_avg = calu_avg_DCG(4577,k)
    print 'DCG_avg : \t' + str(DCG_avg) 

if __name__ == '__main__':
    main()