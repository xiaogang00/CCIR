#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本代码用于对于所有4500个问题进行BM25测试，首先将所有的答案和问题去除stopwords，然后对于出现次数较高的词语进行去除，再求解BM25的值

#sys.agrv[1]为训练数据的路径，sys.argv[2]为训练数据的文件名,sys.argv[3]代表训练数据中的问题的个数，sys.argv[4]代表最后的测试选取的为DCG@k的k值
#sys.argv[5]为设定的单词的数量界限

import json
import sys
import numpy as np
import os
import re
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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

def sort_by_value(d): 
    items=d.items() 
    backitems=[[v[1],v[0]] for v in items] 
    backitems.sort(reverse = True) 
    return [ backitems[i][1] for i in range(0,len(backitems))]
    
def write_words_question(item,word_list,line_num):
    #本函数将问题及答案的所选的词语存入/home/dcd-qa/MLT_code/BM25/data/question_ansers_words_without_stopwords/文件夹中对应的文件内
    file_write = open('/home/dcd-qa/MLT_code/BM25/data/question_ansers_words_without_stopwords/' + str(line_num-1),'ab+')
    file_write.write(item + '\t')
    for word in word_list:
        if word != ' ' and word != '' and word != ' ' and word != '　':
            file_write.write(str(word) + '\t')
    file_write.write('\n')
    file_write.close()
    
def write_words_answer(item,word_list,line_num,label):
    #本函数将问题及答案的所选的词语存入/home/dcd-qa/MLT_code/BM25/data/question_ansers_words_without_stopwords/文件夹中对应的文件内
    file_write = open('/home/dcd-qa/MLT_code/BM25/data/question_ansers_words_without_stopwords/' + str(line_num-1),'ab+')
    file_write.write(item + '\t' + str(label) + '\t')
    for word in word_list:
        if word != ' ' and word != '' and word != ' ' and word != '　':
            file_write.write(str(word) + '\t')
    file_write.write('\n')
    file_write.close()
    
def get_word_dict(train_file,stop_words_list):
    #这个函数用来对于所有的问题及答案的句子进行遍历，首先对于其中的stopwords进行去除，然后找到所有词的出现次数，按照出现次数进行排序之后，对于出现次数特别大的，
    #也进行去除
    f_read = open(train_file,'r')
    r1 = u"[\≤\〔\〕\『\』\\\■\ˋ\ˇ\＋\→\／\～\·\［\］\①\｛\｝\；\％\＝\±\×\’\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\，\。\?\★\、\…\【\】\《\》\？\“\”\‘\’\！\[\]\^\_\`\{\|\}\~\]\+\（\）\~\+\—\：\.\．]"
    word_dict = {}
    line_num = 1
    for line in f_read.readlines():
        if line_num%1000 == 1:
            print 'Now for line_num: ' + str(line_num) + '\n'
        line_num = line_num + 1
        file_dict_temp = json.loads(line)
        query = file_dict_temp['query']
        query = re.sub(r1," " ,query)
        query_word_list = get_words(query)
        #先去除stopwords，并存储下所有单词的出现次数
        for word in query_word_list:
            if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == '　' or word == ' ' or word == u'\u3000':
                query_word_list.remove(word)
            elif word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] = word_dict[word] + 1

        passages_list = file_dict_temp['passages']
        for item in passages_list:
            item_passage_text_temp = item['passage_text']
            item_passage_text_temp_1 = re.sub(r1," " ,item_passage_text_temp)
            item_passage_text_temp_word_list = get_words(item_passage_text_temp_1)
            for word in item_passage_text_temp_word_list:
                if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == '　' or word == ' ' or word == u'\u3000':
                    item_passage_text_temp_word_list.remove(word)
                elif word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] = word_dict[word] + 1
    word_dict_list = sort_by_value(word_dict)
    # for word in word_dict_list:
        # print word,word_dict[word]
    #对于出现次数超过1000次的前103个词，将其从各个question和answer中删去
    word_delete = []
    for word in word_dict_list:
        if word_dict[word] > 1000 or word_dict[word] < 2:
            word_delete.append(word)
    
    f_read.close()
    f_read = open(train_file,'r')
    line_num = 1
    question_length = [] #存储每个question的长度，来得到问题的平均长度
    for line in f_read.readlines():
        if line_num%1000 == 1:
            print 'Now for line_num: ' + str(line_num) + '\n'
        line_num = line_num + 1
        file_dict_temp = json.loads(line)
        query = file_dict_temp['query']
        query = re.sub(r1," " ,query)
        query_word_list = get_words(query)
        #先去除stopwords，并存储下所有单词的出现次数
        for word in query_word_list:
            if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == ' ' or word == '　' or word == u'\u3000' or word in word_delete:
                query_word_list.remove(word)
            else:
                continue
        write_words_question('question',query_word_list,line_num)
        question_length.append(len(query_word_list))
        passages_list = file_dict_temp['passages']
        for item in passages_list:
            item_passage_text_temp = item['passage_text']
            item_label = item['label']
            item_passage_text_temp_1 = re.sub(r1," " ,item_passage_text_temp)
            item_passage_text_temp_word_list = get_words(item_passage_text_temp_1)
            for word in item_passage_text_temp_word_list:
                if word in stop_words_list or word == 'br' or word == ' ' or word == '' or word == ' ' or word == '　' or word == u'\u3000' or word in word_delete:
                    item_passage_text_temp_word_list.remove(word)
                else:
                    continue
            write_words_answer('answer',item_passage_text_temp_word_list,line_num,item_label)
            
    f_read.close()
    return question_length
    
def score_calu_BM25(avg_answer_length,answer_temp_length,answer_word_num_dict,question_line_list,answer_temp_word_num_dict,answer_num):
    question_word_score_list = [] #question中的各个单词的score值
    k1 = 1.5
    b = 0.75
    answer_word_score_list = []
    for word in question_line_list:
        if word == 'question':
            continue
        else:
            if word in answer_word_num_dict:
                n_qi = int(answer_word_num_dict[word])
            else:
                n_qi = 0
            IDF = np.log((int(answer_num) - n_qi + 0.5)/(n_qi + 0.5))
            if word in answer_temp_word_num_dict:
                f_qi_D = int(answer_temp_word_num_dict[word])
            else:
                f_qi_D = 0
            score_temp = (f_qi_D * (k1 + 1))/(f_qi_D + k1 *(1-b+b*(answer_temp_length/avg_answer_length)))
            score_temp = IDF * score_temp
            answer_word_score_list.append(score_temp)
    answer_score = np.sum(answer_word_score_list)
    return answer_score

def calu_avg_answer_length(answer_length):
    length_sum = 0
    for length in answer_length:
        length_sum = length_sum + length
    avg_length = length_sum/len(answer_length)
    return avg_length
    
def calu_answer_score(question_num,word_delete):
    answer_score_label_dir = '/home/dcd-qa/MLT_code/BM25/data/answer_score_label/'
    for id in range(question_num):
        file_name = '/home/dcd-qa/MLT_code/BM25/data/question_ansers_words_without_stopwords/' + str(id+1)
        file_read = open(file_name,'rb')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        for word in question_line_list:
            if word in word_delete:
                question_line_list.remove(word)
        #下面对于该question对应的所有answer进行遍历，得到所有answer中各个词一共出现的次数的dict，为求IDF作准备
        answer_word_num_dict = {}
        answer_length = []
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_length_temp = len(answer_temp_line_list) - 2
            answer_length.append(answer_length_temp)
            word_has_calu = [] #用来记录已经在该答案中检索过的词语，保证每个词语只在一个答案中检索一次
            for index in range(len(answer_temp_line_list)):
                if index < 2:
                    continue
                else:
                    word = answer_temp_line_list[index]
                    if word not in word_has_calu and word not in word_delete:
                        if word not in answer_word_num_dict:
                            answer_word_num_dict[word] = 1
                        else:
                            answer_word_num_dict[word] = answer_word_num_dict[word] + 1
                        word_has_calu.append(word)
                    else:
                        continue
        avg_answer_length = calu_avg_answer_length(answer_length)
        file_read.close()
        file_read = open(file_name,'rb')
        question_line = file_read.readline()
        answer_num = 0
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = answer_temp_line_list[1]
            answer_temp_word_num_dict = {} #存储当前answer的各个词语出现的次数
            for index in range(len(answer_temp_line_list)):
                if index < 2:
                    continue
                else:
                    word = answer_temp_line_list[index]
                    if word not in answer_temp_word_num_dict and word not in word_delete:
                        answer_temp_word_num_dict[word] = 1
                    else:
                        answer_temp_word_num_dict[word] = answer_temp_word_num_dict[word] + 1
            answer_score = score_calu_BM25(avg_answer_length,answer_length[answer_num],answer_word_num_dict,question_line_list,answer_temp_word_num_dict,len(answer_length))
            answer_num = answer_num + 1
            file_write_name = answer_score_label_dir + str(id+1)
            file_write = open(file_write_name,'ab+')
            file_write.write(str(answer_label) + '\t' + str(answer_score))
            file_write.write('\n')
            file_write.close()

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
        if k_temp < k: 
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
    
def get_word_num(question_num,word_num_limit):
    word_num_dict = {}
    for id in range(question_num):
        file_name = '/home/dcd-qa/MLT_code/BM25/data/question_ansers_words_without_stopwords/' + str(id+1)
        file_read = open(file_name,'rb')
        question_line = file_read.readline()
        question_line_list = question_line.strip().split('\t')
        question_line_list.remove('question')
        for word in question_line_list:
            if word not in word_num_dict:
                word_num_dict[word] = 1
            else:
                word_num_dict[word] += 1
        for line in file_read.readlines():
            answer_temp_line_list = line.strip().split('\t')
            answer_label = answer_temp_line_list[1]
            answer_temp_line_list.remove('answer')
            answer_temp_line_list.remove(answer_labe)
            for word in answer_temp_line_list:
                if word not in word_num_dict:
                    word_num_dict[word] = 1
                else:
                    word_num_dict[word] += 1
    word_delete = []
    word_dict_list = sort_by_value(word_num_dict)
    for word in word_dict_list:
        if word_dict[word] > word_num_limit or word_dict[word] < 2:
            word_delete.append(word)
    return word_delete
            
    
def words_statics(question_num,k,word_num_limit):
    #对于train_file中的每一个问题统计一下各个词语出现的频率，并且将stop_words_list中的不纳入统计,对于出现次数特别大的，也进行去除，
    #然后再按照答案的label的高低，对于其中的词去除各自比例，再将剩余的词的出现频率进行存储，存储到一个json文件中，该文件的每一行为一个问题或是一个答案的词语的个数，
    #并且另外存储一个文件来代表各个问题所含有的答案的个数
    #question_length = get_word_dict(train_file,stop_words_list)
    word_delete = get_word_num(question_num,word_num_limit)
    calu_answer_score(question_num,word_delete) #分别计算各个question的所有answer的score值，并且将每个question的score值及对应的label值存入一个文件中
    DCG_result = calu_avg_DCG(len(question_length),k)
    return DCG_result
    
def main():
    train_file_dir = sys.argv[1]
    train_file_name = sys.argv[2]
    question_num = int(sys.argv[3])
    k = int(sys.argv[4])
    word_num_limit = int(sys.argv[5])
    train_file = os.path.join(train_file_dir,train_file_name)
    #stop_words_list = get_stop_words('/home/dcd-qa/MLT_code/BM25/data/chinese_stopwords.txt')
    DCG_result = words_statics(question_num,k,word_num_limit)
    print 'DCG@' + str(k) + '\t' + str(DCG_result) + '\n'
    
    
if __name__ == '__main__':
    main()