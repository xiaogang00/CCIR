#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于对于CCIR的训练数据的问题答案中的中文句子进行单词分解，并对于分解好的单词添加到wiki.zh.text.jian.seg的后面（每一句话一行）

import os
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json

#sys.argv[1]为训练文件的路径
#sys.argv[2]为要存储所有训练数据的句子中的单词的文件路径
#sys.argv[3]为训练文件的名字
#sys.argv[4]为存储无法分词的问题的query_id的文件

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
    
    
def write_sentence_word(write_file_name,sentence_list):
    file_write = open(write_file_name,'ab+')
    for word in sentence_list:
        if word != 'br' and word != '<' and word != '>' :
            file_write.write(str(word)+'\t')
    file_write.write('\n')
    print str(sentence_list) + 'done'
    
def process_file(write_file_name):
    file_read = open(write_file_name,'rb')
    for line in file_read.readlines():
        line_new = ' '.join(filter(lambda x: x, line.split('\t')))
        line_new = ' '.join(filter(lambda x: x, line_new.split(' ')))
        file_write = open(write_file_name+'_new','ab+')
        file_write.write(line_new.strip() + '\n')
        file_write.close()
    
def process_train_file(train_file_dir,train_file_name,write_file_name,error_file_path):
    #本程序用来处理训练文件，对于其中的每个问题的句子进行分割，构造query的词语构成的list，及答案的词语构成的list，将这些list的元素按照每句话的所有单词一行存储进入到一个文件train_setences中
    f = open(os.path.join(train_file_dir,train_file_name),'r')
    r1 = u"[\⊙\【\】\€發\Q\Cr\C\┙\τ\o\ζ\π\┮\∷\�\←\☆\≤\〔\〕\『\』\\\■\ˋ\ˇ\＋\→\／\～\·\［\］\①\｛\｝\；\％\＝\±\×\’\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\，\。\?\★\、\…\【\】\《\》\？\“\”\‘\’\！\[\]\^\_\`\{\|\}\~\]\+\（\）\~\+\—\：\.\．]"
    for line in f.readlines():
        file_dict_temp = json.loads(line)
        query = file_dict_temp['query']
        query_id = file_dict_temp['query_id']
        query = re.sub(r1," " ,query)
        #query = query.encode('utf-8')
        query_word_list = get_words(query,query_id)
        if query_word_list[0] == 'ERROR':
            error_file = open(error_file_path,'ab+')
            error_file.write(str(query_word_list[1]) + '\n')
            error_file.close()
            continue
        write_sentence_word(write_file_name,query_word_list)
        passages_list = file_dict_temp['passages']
        for item in passages_list:
            item_passage_text_temp = item['passage_text']
            item_passage_text_temp_1 = re.sub(r1," " ,item_passage_text_temp)
            #item_passage_text_temp_1 = item_passage_text_temp_1.encode('utf-8')
            #print item_passage_text_temp_1.__class__
            #print 'item_passage_text_temp_1 : ' + str(item_passage_text_temp_1)
            item_passage_text_temp_word_list = get_words(item_passage_text_temp_1)
            if item_passage_text_temp_word_list[][0] == 'ERROR':
                error_file = open(error_file_path,'ab+')
                error_file.write(str(query_word_list[1]) + '\n')
                error_file.close()
                break
            write_sentence_word(write_file_name,item_passage_text_temp_word_list)
    #随后对于文件中的多于的'\t'全部合并成为一个，并且对于'\t'替换成为' '
    process_file(write_file_name)
    
def main():
    train_file_dir = sys.argv[1] 
    train_file_name = sys.argv[3]
    write_file_name = sys.argv[2]
    error_file_path = sys.argv[4]
    process_train_file(train_file_dir,train_file_name,write_file_name,error_file_path)
    
if __name__ == '__main__':
    main()