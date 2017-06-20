#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于合并关于wordoverlap和CNN的结果，首先对于词频进行处理，对于词频大于m的单词就不再计入关于相同词数的判断了，
#随后对于答案长度进行判断，对于答案长度大于320的就不使用CNN的结果了，之后对于相同词的数目进行判断，对于相同词数目大于n的，就用wordoverlap进行判断，得到相关的wordoverlap的得分
#对于相同词数目小于n的，直接使用CNN的得分

#sys.argv[1]为wordoverlap所分解出的问题与答案的单词文件的目录，sys.argv[2]为测试数据中共有的问题数目(包括无法分词的问题)
#sys.argv[3]为近义词的存储位置，sys.argv[4]为计入相同词的词频界限m，sys.argv[5]为利用wordoverlap进行score判断的词数界限n，sys.argv[6]为wordoverlap的score存储目录的前缀
#sys.argv[7]为CNN的分数文件位置，sys.argv[8]为DCG的k值

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
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
    #DCG的计算函数
    answer_index_label_dict = {} #存储answer_index与label的对应dict
    answer_index_score_dict = {} #存储answer_index与score的对应dict
    #score_answer_index_dict = {} #存储score与answer_index的对应dict
    answer_index = 0
    for line in open(answer_score_file,'rb').readlines():
        line_list = line.strip().split()
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

def get_delete_word(same_word_limit):
    file_read = open('/home/duanxinyu/MLT/CCIR/MLT_code_test/word2vec/CCIR_train_test_word_num.txt','rb')
    #file_read = open('D:\\CCIR\\code_for_json3\\wordoverlap_file\\CCIR_train_3_word_num.txt','rb')
    word_delete = []
    for line in file_read.readlines():
        line = line.strip()
        if line == '':
            continue
        else:
            line_list = line.split('\t')
            word_temp = line_list[0]
            word_num = int(line_list[1])
            if word_num > same_word_limit:
                word_delete.append(str(word_temp))
    return word_delete

def get_file_wordoverlap_score(wordoverlap_file_path,wordoverlap_new_dir,index,synonym_dict_temp,delete_word_list):
    file_read = open(wordoverlap_file_path,'rb+')
    question_line = file_read.readline()
    question_line_list = question_line.strip().split('\t')
    question_line_list.remove('question')
    file_write_name = os.path.join(wordoverlap_new_dir,str(index)) #test的question的id从0开始计数
    file_write = open(file_write_name,'ab+')
    for line in file_read.readlines():
        answer_temp_line_list = line.strip().split('\t')
        answer_label = answer_temp_line_list[1]
        answer_temp_line_list.remove('answer')
        answer_temp_line_list.remove(answer_label)
        answer_temp_same_word_num = 0
        for word_1 in question_line_list:
            for word_2 in answer_temp_line_list:
                if word_1 in delete_word_list or word_2 in delete_word_list:
                    continue
                elif str(word_1) == str(word_2):
                    answer_temp_same_word_num += 1
                    continue
                elif str(word_1) in synonym_dict_temp and str(word_2) in synonym_dict_temp[str(word_1)]: 
                    answer_temp_same_word_num += 1
        file_write.write(answer_label + '\t' + str(answer_temp_same_word_num))
        file_write.write('\n')
    file_write.close()
                    
    
def get_wordoverlap_file_score(wordoverlap_file_dir,question_num,synonym_dir,wordoverlap_limit,wordoverlap_new_dir,delete_word_list):
    for index in range(question_num):
        if (index + 1)%1000 == 1:
            print 'Now for line : ' + str(index+1) + '\n'
        same_word_num = 0
        wordoverlap_file_path = os.path.join(wordoverlap_file_dir,str(index+1))
        synonym_file_name = os.path.join(synonym_dir,str(index+1))
        if not os.path.isfile(synonym_file_name) or not os.path.join(wordoverlap_file_path):
            continue
        file_synonym_read = open(synonym_file_name,'rb')
        synonym_dict_temp = {}
        for line in file_synonym_read.readlines():
            line_list = line.strip().split('\t')
            word_q = line_list[0]
            word_w = line_list[1]
            if str(word_q) not in synonym_dict_temp:
                synonym_dict_temp[str(word_q)] = []
            synonym_dict_temp[str(word_q)].append(str(word_w)) #对于从问题到答案的近义词，有问题的同一个词与答案中的多个词均为近义词的情况
        file_read = open(wordoverlap_file_path,'rb+')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        question_line_list.remove('question')
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = answer_temp_line_list[1]
            answer_temp_line_list.remove('answer')
            answer_temp_line_list.remove(answer_label)
            for word_1 in question_line_list:
                for word_2 in answer_temp_line_list:
                    if word_1 in delete_word_list or word_2 in delete_word_list:
                        continue
                    elif str(word_1) == str(word_2):
                        same_word_num += 1
                        continue
                    elif str(word_1) in synonym_dict_temp and str(word_2) in synonym_dict_temp[str(word_1)]: #可能question的word_1没有在近义词表中出现
                        same_word_num += 1
        if same_word_num < wordoverlap_limit:
            continue
        else:
            get_file_wordoverlap_score(wordoverlap_file_path,wordoverlap_new_dir,index,synonym_dict_temp,delete_word_list)
            
def get_DCG_final_score(wordoverlap_new_dir,CNN_file_dir,question_num,k):
    #对于wordoverlap_new_dir含有的分数，按照wordoverlap的分数处理，对于wordoverlap_new_dir没有的分数，按照CNN的分数来处理
    DCG_score_list = []
    for index in range(question_num):
        file_wordoverlap_path = os.path.join(wordoverlap_new_dir,str(index))
        file_CNN_path = os.path.join(CNN_file_dir,str(index))
        if os.path.isfile(file_wordoverlap_path):
            DCG_score_temp = calu_DCG(file_wordoverlap_path,k)
            DCG_score_list.append(DCG_score_temp)
        else:
            DCG_score_temp = calu_DCG(file_CNN_path,k)
            DCG_score_list.append(DCG_score_temp)
    DCG_avg = calu_avg_answer_length(DCG_score_list)
    return DCG_avg
            
def main():
    wordoverlap_file_dir = sys.argv[1]
    question_num = int(sys.argv[2])
    synonym_dir = sys.argv[3]
    same_word_limit = int(sys.argv[4])
    wordoverlap_limit = int(sys.argv[5])
    wordoverlap_new_dir = sys.argv[6]
    CNN_file_dir = sys.argv[7]
    k = int(sys.argv[8])
    delete_word_list = get_delete_word(same_word_limit)
    get_wordoverlap_file_score(wordoverlap_file_dir,question_num,synonym_dir,wordoverlap_limit,wordoverlap_new_dir,delete_word_list)
    #DCG_avg = get_DCG_final_score(wordoverlap_new_dir,CNN_file_dir,question_num,k)
    #print 'DCG@' + str(k) + ' of same_word_limit:' + str(same_word_limit) + ' wordoverlap_limit:' + str(wordoverlap_limit) + ' is : ' + str(DCG_avg) + '\n\n'
    
if __name__ == '__main__':
    main()