#! /usr/env/bin python
# -*-coding: utf-8 -*-

import pynlpir
import json
import os
import re


def get_words(sentence):
    # 输入为一个string的句子，输出为这个句子的分解的单词
    pynlpir.open()
    sentence_words_list = pynlpir.segment(sentence, pos_tagging=False)
    return sentence_words_list
    
def process_train_file(train_file_dir,train_file_name):
    # 本程序用来处理训练文件，对于其中的每个问题的句子进行分割，构造query的词语构成的list，及答案的词语构成的list
    # train_file_dir为训练文件的路径，train_file_name为训练文件的名字
    f = open(os.path.join(train_file_dir,train_file_name),'r')
    r1 = u"[a-zA-Z0-9\σ\δ\≤\〔\〕\『\』\\\ｔ\Ｔ\Ｌ\■\Ｆ\ａ\ｌ\ｓ\ｈ\ㄅ\ㄨ\ˋ\ㄧ\ㄡ\ㄗ\ㄓ\ˇ\ｒ\ｏ\ｔ\ｉ\ｎ\ｇ\ｕ\ｅ\Ｒ\Ｏ\Ｔ\Ｉ\Ｎ\Ｇ\＋\→\／\～\β\Ｍ\Ｖ\Ｐ\Ｎ\Ｂ\Ａ\Ｃ\ｖ\Ｄ\Ｌ\Ｗ\Ｇ\ɡ\ɑ\Ｑ\ｄ\·\［\］\à\①\ī\é\í\ǔ\ì\ó\ù\｛\｝\；\１\８\ｋ\７\５\％\２\Ｋ\０\４\＝\６\±\×\’\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\，\。\?\★\、\…\【\】\《\》\？\“\”\‘\’\！\[\]\^\_\`\{\|\}\~\]\+\（\）\~\+\—\：\.\．]"
    question = []
    answer = []
    count = 0
    for line in f:
        file_dict_temp = json.loads(line)
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
    train_file_name = 'train.1.json'
    [q, item] = process_train_file(train_file_dir, train_file_name)
    print len(q)
    print len(item[0][1])

