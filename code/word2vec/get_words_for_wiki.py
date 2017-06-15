#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本程序用于对于wiki的数据进行分词，并将每一行的分词用空格分隔，去除其中的英文，存储到一个文件中

import pynlpir

def get_words(sentence):
    #输入为一个string的句子，输出为这个句子的分解的单词
    pynlpir.open()
    try:
        sentence_words_list = pynlpir.segment(sentence,pos_tagging=False)
    except UnicodeDecodeError:
        return []
    return sentence_words_list
    
def main():
    file_wiki_simple_read = open('/home/dcd-qa/MLT_code/CCIR_code/wiki.zh.text.jian','rb')
    alphbet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    line_num = 1
    for line in file_wiki_simple_read.readlines():
        if line_num%10000 == 1:
            print 'Now for line : ' + str(line_num) + '\n'
        line_num = line_num + 1
        for letter in alphbet:
            line = line.replace(letter,'')
        line_word_list = get_words(line)
        file_write = open('./wiki.zh.text.jian.seg','ab+')
        for word in line_word_list:
            try:
                file_write.write(str(word.encode('utf-8')) + ' ')
            except UnicodeDecodeError:
                continue
        file_write.write('\n')
        file_write.close()

if __name__ == '__main__':
    main()