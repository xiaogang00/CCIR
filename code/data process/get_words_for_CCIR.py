#! /usr/env/bin python
# -*-coding: utf-8 -*-

# 本程序是从json文件当中读取我们所需要的问题，答案以及其对应的queryid，在其中对于不能分词的情况选择跳过
import pynlpir
import json
import os
import re
import string

def get_words(sentence):
    # 输入为一个string的句子，输出为这个句子的分解的单词
    pynlpir.open()
    sentence_words_list = pynlpir.segment(sentence, pos_tagging=False)
    return sentence_words_list

# def process_id(train_file_dir,train_file_name, unable_file_dir, unable_file_name):
def process_id(train_file_dir,train_file_name):
    # 本程序用来处理训练文件，对于其中的每个问题的句子进行分割，构造query的词语构成的list，及答案的词语构成的list
    # train_file_dir为训练文件的路径，train_file_name为训练文件的名字
    unable = open(os.path.join(train_file_dir, 'cannot_segment_query_id.txt'), 'r')
    lines = unable.readlines()
    unable_id = []
    for line in lines:
        a = line.replace("\n", "").split("	")
        unable_id.append(string.atoi(a[0]))
    f = open(os.path.join(train_file_dir, train_file_name),'r')
    qaid = []
    for line in f:
        file_dict_temp = json.loads(line)
        temp_id = file_dict_temp['query_id']
        if temp_id in unable_id:
            continue
        qaid.append(temp_id)
    return qaid


def process_train_file(train_file_dir,train_file_name):
    # 本程序用来处理训练文件，对于其中的每个问题的句子进行分割，构造query的词语构成的list，及答案的词语构成的list
    # train_file_dir为训练文件的路径，train_file_name为训练文件的名字
    f = open(os.path.join(train_file_dir,train_file_name),'r')
    unable = open(os.path.join(train_file_dir, 'cannot_segment_query_id.txt'), 'r')
    r1 = u"[\⊙\【\】\€發\Q\Cr\C\┙\τ\o\ζ\π\┮\∷\�\←\☆\≤\〔\〕\『\』\\\■\ˋ\ˇ\＋\→\／\～\·\［\］\①\｛\｝\；\％\＝\±\×\’\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\，\。\?\★\、\…\【\】\《\》\？\“\”\‘\’\！\[\]\^\_\`\{\|\}\~\]\+\（\）\~\+\—\：\.\．]"
    question = []
    answer = []
    count = 0
    lines = unable.readlines()
    unable_id = []
    for line in lines:
        a = line.replace("\n", "").split("	")
        unable_id.append(string.atoi(a[0]))
    # print unable_id
    for line in f:
        file_dict_temp = json.loads(line)
        temp_id = file_dict_temp['query_id']
        if temp_id in unable_id:
            continue
        query = file_dict_temp['query']
        query = re.sub(r1," " ,query)
        query_word_list = get_words(query)
        question.append(query_word_list)
        # for i in range(len(query_word_list)):
        #    print query_word_list[i],
        # print '\n'
        #query_word_list中为问题中的词语（去除其中的r1中的特殊符号）
        passages_list = file_dict_temp['passages']
        answer.append([])
        for item in passages_list:
            item_passage_text_temp = item['passage_text']
            item_passage_text_temp_1 = re.sub(r1," " ,item_passage_text_temp)
            item_passage_text_temp_word_list = get_words(item_passage_text_temp_1)
            answer[count].append(item_passage_text_temp_word_list)
            # for i in range(len(item_passage_text_temp_word_list)):
            #    print item_passage_text_temp_word_list[i],
            # print '\n'
        count = count + 1
    return question, answer
            #item_passage_text_temp_word_list为每个答案中的单词（去除其中的r1中的特殊符号）

if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'json.longshort.new.all'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    print q
    print len(item[0])

