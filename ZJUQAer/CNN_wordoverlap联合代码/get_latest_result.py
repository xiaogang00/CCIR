#! /usr/env/bin python
# -*-coding: utf-8 -*-

#本代码用于生成最终符合要求的测试结果，对于score值的文件读入之后，再排序，将score值高的答案的passage_id放在形成的dict['ranklist']前面，最后再将形成的dict存储为一个json文件
#sys.argv[1]为最终的得到存储新的wordoverlap的score值的文件夹(文件的名字为query_id，对于无法分词的question，人工将答案排好顺序，并且将自己设定的分数（更好的答案给更高的分数）写入一个以query_id命名的文件中，将这些文件与新的wordoverlap形成的分数文件都放在这个文件夹里)，各个score值对应与于最终的各个答案的顺序，sys.argv[2]为含有的question的query_id的最大数目(包括无法分词的question)
#sys.argv[3]为最终给的测试json文件，sys.argv[4]为最终形成的result的json文件的路径
#sys.argv[5]为存储CNN的结果文件夹


import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import json
import codecs
import os
import numpy as np

def sort_by_value(d): 
    items=d.items() 
    backitems=[[v[1],v[0]] for v in items] 
    backitems.sort(reverse = True) 
    return [ backitems[i][1] for i in range(0,len(backitems))]

def write_get_result_file(query_id,query,score_file_path,result_json_path):
    file_read = open(score_file_path,'rb')
    score_index_num = {}
    score_index = 1
    score_list = []
    for line in file_read.readlines():
        score_temp = float(line.strip())
        score_index_num[score_index] = score_temp
        score_list.append(score_temp)
        score_index += 1
    score_num = len(score_list)
    if np.std(score_list) < 1:
        score_list = np.random.rand(score_num)*2
        score_index = 1
        for item in score_list:
            score_index_num[score_index] = item
            score_index += 1
    score_index_sorted_list = sort_by_value(score_index_num)
    result_dict = {}
    result_dict["query_id"] = int(query_id)
    result_dict["query"] = str(query)
    result_dict["ranklist"] = []
    rank_index = 0
    for score_index in score_index_sorted_list:
        #对于排序靠前的answer存储相关信息到要收集的结果dict中
        dict_temp = {}
        dict_temp["passage_id"] = int(score_index) #score_index即为原始的score值对应的answer的序列号
        dict_temp["rank"] = int(rank_index)
        rank_index += 1
        result_dict["ranklist"].append(dict_temp)

    # file_write = open(result_json_path,'ab+')
    file_write = codecs.open(result_json_path, 'ab+', 'utf-8')
    #file_write.write('{\n' + '\"query_id\":' + str(a["query_id"]) + ',\n\"query\":'+str(a["query"])+ ',\n\"ranklist\":'+str(a["ranklist"]) + '\n}')
    file_write.write(json.dumps(result_dict, ensure_ascii=False))
    file_write.write('\n')
    file_write.close()

def get_result_json_file(answer_score_dir,test_question_num,test_json_file_path,result_json_path,CNN_result_dir):
    line_index = 0
    file_read = open(test_json_file_path,'r')
    for line in file_read.readlines():
        if (line_index+1)%1000 == 1:
            print 'Now for line : ' + str(line_index + 1) + '\n'
        line_index += 1
        file_dict_temp = json.loads(line)
        query_id = file_dict_temp['query_id']
        query = file_dict_temp['query']
        answer_score_file_path = os.path.join(answer_score_dir,str(query_id))
        CNN_score_file_path = os.path.join(CNN_result_dir,str(query_id))
        if os.path.isfile(answer_score_file_path):
            #则score值的文件在answer_score_dir这个目录中，那么就对于该文件夹下面的answer_score_file_path文件进行处理，得到result文件的一项
            write_get_result_file(query_id,query,answer_score_file_path,result_json_path)
        else:
            write_get_result_file(query_id,query,CNN_score_file_path,result_json_path)
            
        

def main():
    answer_score_dir = sys.argv[1]
    test_question_num = sys.argv[2]
    test_json_file_path = sys.argv[3]
    result_json_path = sys.argv[4]
    CNN_result_dir = sys.argv[5]
    get_result_json_file(answer_score_dir,test_question_num,test_json_file_path,result_json_path,CNN_result_dir)
    
if __name__ == '__main__':
    main()