from get_words_for_CCIR import *
import numpy as np

def frequency_word(train_file_dir,train_file_name):
    f = open(os.path.join(train_file_dir, train_file_name), 'r')
    lines = f.readlines()
    word = []
    count = 0
    for line in lines:
        a = line.replace("\n","").split("	")
        if len(a) == 1:
            continue
        if int(a[1]) < 50:
            continue
        word.append(a[0])
        count = count + 1
    #for i in range(len(word)):
    #    print word[i]
    return word




if __name__ == '__main__':
    train_file_dir = './'
    train_file_name = 'CCIR_train_word_num.txt'
    frequency(train_file_dir, train_file_name)


