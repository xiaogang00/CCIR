#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于对于CCIR的问题及答案的语句进行分词，分词的结果放到wiki的数据后面
#sys.agrv[1]为训练数据的路径，sys.argv[2]为训练数据的文件名

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import os
import re
import numpy as np

import pynlpir

def get_words(sentence):
    #输入为一个string的句子，输出为这个句子的分解的单词
    pynlpir.open()
    sentence_words_list = pynlpir.segment(sentence,pos_tagging=False)
    return sentence_words_list

def get_stop_words(file_path):
    file_read = open(file_path,'rb')
    stop_words_list = []
    for line in file_read.readlines():
        word = line.strip()
        stop_words_list.append(word)
    return stop_words_list
    
def write_file(query_word_list):
    file_write = open('./wiki.zh.text.jian.seg','ab+')
    for word in query_word_list:
        if word != ' ' and word != '' and word != ' ' and word != '　':
            file_write.write(' ' + str(word))
    file_write.write('\n')
    file_write.close()
    
def get_question_answer_word(train_file,stop_words_list):
    #这个函数用来对于所有的问题及答案的句子进行遍历，首先对于其中的stopwords进行去除，然后找到所有词的出现次数，按照出现次数进行排序之后，对于出现次数特别大的，
    #也进行去除
    f_read = open(train_file,'r')
    r1 = u"[\≤\〔\〕\『\』\\\■\ˋ\ˇ\＋\→\／\～\·\［\］\①\｛\｝\；\％\＝\±\×\’\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\，\。\?\★\、\…\【\】\《\》\？\“\”\‘\’\！\[\]\^\_\`\{\|\}\~\]\+\（\）\~\+\—\：\.\．]"
    line_num = 1
    question_num = 0
    for line in f_read.readlines():
        if line_num%1000 == 1:
            print 'Now for line_num: ' + str(line_num) + '\n'
        line_num = line_num + 1
        file_dict_temp = json.loads(line)
        query = file_dict_temp['query']
        query = re.sub(r1," " ,query)
        query_word_list = get_words(query)
        #先去除stopwords，并存储下该问题的所有单词
        for word in query_word_list:
            if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == ' ' or word == '　' or word == u'\u3000':
                query_word_list.remove(word)
        write_file(query_word_list)
        passages_list = file_dict_temp['passages']
        for item in passages_list:
            item_passage_text_temp = item['passage_text']
            item_passage_text_temp_1 = re.sub(r1," " ,item_passage_text_temp)
            item_passage_text_temp_word_list = get_words(item_passage_text_temp_1)
            for word in item_passage_text_temp_word_list:
                if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == ' ' or word == '　' or word == u'\u3000':
                    item_passage_text_temp_word_list.remove(word)
            write_file(item_passage_text_temp_word_list)
            
def process_file(write_file_name):
    file_read = open(write_file_name,'rb')
    for line in file_read.readlines():
        line_new = ' '.join(filter(lambda x: x, line.split('\t')))
        line_new = ' '.join(filter(lambda x: x, line_new.split(' ')))
        file_write = open(write_file_name+'_new','ab+')
        file_write.write(line_new.strip() + '\n')
        file_write.close()
        
    
def main():
    train_file_dir = sys.argv[1]
    train_file_name = sys.argv[2]
    train_file = os.path.join(train_file_dir,train_file_name)
    stop_words_list = get_stop_words('/home/dcd-qa/MLT_code/NLPCC/data/chinese_stopwords.txt')
    get_question_answer_word(train_file,stop_words_list)
    process_file('./wiki.zh.text.jian.seg')
    
if __name__ == '__main__':
    main()