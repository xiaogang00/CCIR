#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于循环执行bm25的程序，来求解每个不同的参数条件下的BM25的值

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import commands

def main():
    word_num_limit_list = [1000,500,200,100,50,40,30]
    k_list = [3,5]
    train_dir = '/home/dcd-qa/MLT_code_2/word2vec/'
    train_file_name = 'train.2.json'
    question_num = '28860'
    for word_num_limit in word_num_limit_list:
        for k in k_list:
            word_num_limit = str(word_num_limit)
            k = str(k)
            print 'Now for k : ' + str(k) + ' word_num_limit : ' + str(word_num_limit)
            commands.getstatusoutput("python get_bm25.py " + train_dir + " " + train_file_name + " " + question_num + " " + k + " " + word_num_limit)

if __name__ == '__main__':
    main()