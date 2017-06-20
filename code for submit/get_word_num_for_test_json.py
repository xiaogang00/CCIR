#! /usr/env/bin python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os

#sys.argv[1]为wordoverlap分解好的问题与答案的单词的文件路径，sys.argv[2]为总共的问题的个数
#wordoverlap已经把stopwords去除掉了

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]
    
def get_word_num(train_file_dir,question_num):
    word_dict = {}
    for index in range(question_num):
        if (index+1)%1000 == 1:
            print 'Now for file : ' + str(index+1)
        file_name = os.path.join(train_file_dir,str(index))
        if not os.path.isfile(file_name):
            continue
        else:
            file_read = open(file_name,'rb')
            question_line = file_read.readline()
            question_line_list = question_line.strip().split('\t')
            question_line_list.remove('question')
            for word in question_line_list:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
            for line in file_read.readlines():
                answer_line_list = line.strip().split('\t')
                answer_label = answer_line_list[1]
                answer_line_list.remove('answer')
                answer_line_list.remove(answer_label)
                for word in question_line_list:
                    if word not in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1
    word_dict_list = sort_by_value(word_dict)
    file_write = open('./CCIR_test_word_num.txt','ab+')
    for word in word_dict_list:
        file_write.write(word + '\t' + str(word_dict[word]))
        file_write.write('\n')
        file_write.write('\n')
    file_write.close()
    
def main():
    train_file_dir = sys.argv[1]
    question_num = sys.argv[2]
    get_word_num(train_file_dir,int(question_num))
    
if __name__ == '__main__':
    main()