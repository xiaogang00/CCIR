#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本代码用于实现NLPCC的第一个baseline，wordoverlap，计算question与answer的共同含有的词数，然后对于每个answer确定其score值（即为重复的元素的个数），按照score的值和
#label值存储之后，便可执行相应的calu_DCG函数计算结果了

#sys.agrv[1]为训练数据的路径，sys.argv[2]为训练数据的文件名

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import os
import re
import numpy as np

import pynlpir

def get_words(sentence,query_id):
    #输入为一个string的句子，输出为这个句子的分解的单词
    pynlpir.open()
    print 'sentence : ' + str(sentence)
    try:
        sentence_words_list = pynlpir.segment(sentence,pos_tagging=False)
        return sentence_words_list
    except BaseException:
        return ['ERROR',str(query_id)]

def get_stop_words(file_path):
    file_read = open(file_path,'rb')
    stop_words_list = []
    for line in file_read.readlines():
        word = line.strip()
        stop_words_list.append(word)
    return stop_words_list
    
def write_words_question(item,word_list,query_id):
    #本函数将问题及答案的所选的词语存入'/home/dcd-qa/MLT_code/NLPCC/data/wordoverlap/question_ansers_words_without_stopwords/'文件夹中对应的文件内
    file_write = open('/home/duanxinyu/MLT/CCIR/MLT_code_test/NLPCC/data/wordoverlap/question_ansers_words_without_stopwords/' + str(query_id),'ab+')
    file_write.write(item + '\t')
    for word in word_list:
        if word != ' ' and word != '' and word != ' ' and word != '　':
            file_write.write(str(word) + '\t')
    file_write.write('\n')
    file_write.close()
    
def write_words_answer(item,word_list,query_id,label):
    #本函数将问题及答案的所选的词语存入'/home/dcd-qa/MLT_code/NLPCC/data/wordoverlap/question_ansers_words_without_stopwords/'文件夹中对应的文件内
    file_write = open('/home/duanxinyu/MLT/CCIR/MLT_code_test/NLPCC/data/wordoverlap/question_ansers_words_without_stopwords/' + str(query_id),'ab+')
    file_write.write(item + '\t' + str(label) + '\t')
    for word in word_list:
        if word != ' ' and word != '' and word != ' ' and word != '　':
            file_write.write(str(word) + '\t')
    file_write.write('\n')
    file_write.close()
    
def get_question_answer_word(train_file,stop_words_list):
    #这个函数用来对于所有的问题及答案的句子进行遍历，首先对于其中的stopwords进行去除，然后存储该问题和答案的单词进入一个文件中
    f_read = open(train_file,'r')
    r1 = u"[\【\】\€發\Q\Cr\C\┙\τ\o\ζ\π\┮\∷\�\←\☆\≤\〔\〕\『\』\\\■\ˋ\ˇ\＋\→\／\～\·\［\］\①\｛\｝\；\％\＝\±\×\’\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\，\。\?\★\、\…\【\】\《\》\？\“\”\‘\’\！\[\]\^\_\`\{\|\}\~\]\+\（\）\~\+\—\：\.\．]"
    line_num = 1
    question_num = 0
    for line in f_read.readlines():
        if line_num%1000 == 1:
            print 'Now for line_num: ' + str(line_num) + '\n'
        line_num = line_num + 1
        file_dict_temp = json.loads(line)
        query = file_dict_temp['query']
        query_id = file_dict_temp['query_id']
        query = re.sub(r1," " ,query)
        query_word_list = get_words(query,query_id)
        if query_word_list[0] == 'ERROR':
            continue
        #先去除stopwords，并存储下该问题的所有单词
        for word in query_word_list:
            if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == ' ' or word == '　' or word == u'\u3000':
                query_word_list.remove(word)
        write_words_question('question',query_word_list,query_id)
        question_num = question_num + 1
        passages_list = file_dict_temp['passages']
        for item in passages_list:
            item_passage_text_temp = item['passage_text']
            item_passage_text_temp_1 = re.sub(r1," " ,item_passage_text_temp)
            item_passage_text_temp_word_list = get_words(query,query_id)
            if query_word_list[0] == 'ERROR':
                break
            item_label = item['label']
            for word in item_passage_text_temp_word_list:
                if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == ' ' or word == '　' or word == u'\u3000':
                    item_passage_text_temp_word_list.remove(word)
            write_words_answer('answer',item_passage_text_temp_word_list,query_id,item_label)
    return question_num
    
def get_score(question_num):
    for index in range(question_num):
        question_index = index + 1
        file_read = open('/home/duanxinyu/MLT/CCIR/MLT_code_3/NLPCC/data/wordoverlap/question_ansers_words_without_stopwords/' + str(question_index),'rb')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        question_line_list.remove('question')
        answer_line_list = []
        file_write = open('/home/duanxinyu/MLT/CCIR/MLT_code_3/NLPCC/data/wordoverlap/answer_label_score/' + str(question_index),'wb')
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = answer_temp_line_list[1]
            answer_temp_line_list.remove('answer')
            answer_temp_line_list.remove(answer_label)
            answer_temp_score = 0
            for word in question_line_list:
                if word in answer_temp_line_list:
                    answer_temp_score = answer_temp_score + 1
                else:
                    continue
            file_write.write(answer_label + '\t' + str(answer_temp_score))
            file_write.write('\n')
        file_write.close()

        
    
def main():
    train_file_dir = sys.argv[1]
    train_file_name = sys.argv[2]
    train_file = os.path.join(train_file_dir,train_file_name)
    stop_words_list = get_stop_words('/home/duanxinyu/MLT/CCIR/MLT_code_3/chinese_stopwords.txt')
    question_num = get_question_answer_word(train_file,stop_words_list)
    print 'question_num : ' + str(question_num)
    #get_score(question_num)
    
if __name__ == '__main__':
    main()