import os
if __name__ == '__main__':
    if not os.path.exists('./test'):
        os.makedirs('./test')
    file_object = open('./test/%d' %(1), 'w')
    a = "answer number %d  %d  %lf  \n" % (1, 2, 0.8)
    file_object.write(a)
    a = "answer number %d  %d  %lf  \n" % (1, 2, 0.8)
    file_object.write(a)