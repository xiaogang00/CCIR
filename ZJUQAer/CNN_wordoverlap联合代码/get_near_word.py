#! /usr/env/bin python
# -*-coding: utf-8 -*-

#该程序用于构建近义词表，并把每个问题的近义词表存入一个文件中
#sys.argv[4](为1个0~1之间的数)用于输入余弦距离大于多大比例的余弦距离才会被定义为近义词（余弦距离越大，代表两个向量越相似），sys.argv[5]用于定义最后的近义词会被输入到哪个目录中，近义词表的文件名也为对应的query_id……
#sys.argv[1]是wordoverlap的分词的目录，sys.argv[2]是总的问题的query最大的数目(包括无法分词的问题)，sys.argv[3]是word2vec的model的目录

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import gensim

def calu_cosin_num(word_q,word_a,model):
    #print 'word_q : ' + str(word_q) + '\n'
    #print 'word_a : ' + str(word_a) + '\n'
    try:
        q_vector = model[word_q.decode('utf-8')]
        a_vector = model[word_a.decode('utf-8')]
    except KeyError:
        return 0
    cosine_similarity_num = cosine_similarity(np.array(q_vector).reshape(1,-1),np.array(a_vector).reshape(1,-1))
    return float(cosine_similarity_num)

def get_synonym_file(question_file_dir,question_num,model,cosine_min_percentage,synonym_dir):
    for question_index in xrange(question_num):
        if (question_index+1)%1000 == 1:
            print 'Now get synonym for line : ' + str(question_index+1) + '\n'
        file_read_name = os.path.join(question_file_dir,str(question_index))
        if not os.path.isfile(file_read_name):
            continue
        file_read = open(file_read_name,'rb+')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        question_line_list.remove('question')
        synonym_index = 0 #用来指示近义词及相似度的index
        cosine_num_dict = {} #对于每一个question及其所有的answer构建了一个dict,代表question中的每个词与answer中的每个词之间的余弦距离
        cosine_num_list = []
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            #answer_label = answer_temp_line_list[1]
            answer_temp_line_list.remove('answer')
            #answer_temp_line_list.remove(answer_label)
            for word_q in question_line_list:
                for word_a in answer_temp_line_list:
                    if (str(word_q),str(word_a)) not in cosine_num_dict:
                        cosine_num = calu_cosin_num(word_q,word_a,model)
                        cosine_tuple = (str(word_q),str(word_a))
                        cosine_num_dict[cosine_tuple] = cosine_num
                        cosine_num_list.append(cosine_num)
       #下面对于cosine_num_list中的所有余弦距离值进行排序，找到对于当前question的余弦距离的分界标准
        cosine_num_list_sorted = sorted(cosine_num_list)
        k = int(len(cosine_num_list)*cosine_min_percentage)
        cosine_num_limit = cosine_num_list_sorted[k]
        file_write = open(os.path.join(synonym_dir,str(question_index)),'ab+')
        for cosine_tuple in cosine_num_dict:
            #print "question_word : " + str(cosine_tuple[0]) + '\n'
            #print "answer_word : " + str(cosine_tuple[1]) + '\n'
            if cosine_num_dict[cosine_tuple] > cosine_num_limit:
                file_write.write(cosine_tuple[0] + '\t' + cosine_tuple[1] + '\n')
            else:
                continue
        file_write.close()
        

def main():
    question_file_dir = sys.argv[1]
    question_num = int(sys.argv[2])
    gensim_model_dir = sys.argv[3]
    gensim_model = os.path.join(gensim_model_dir,'wiki.zh.text.model')
    model = gensim.models.Word2Vec.load(gensim_model)
    cosine_min_percentage = float(sys.argv[4])
    synonym_dir = sys.argv[5]
    get_synonym_file(question_file_dir,question_num,model,cosine_min_percentage,synonym_dir)
    
if __name__ == '__main__':
    main()