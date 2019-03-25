#coding:utf-8
"""
license for studing and Non-profit organization and group.
all right reserved by adau22@163.com,owned by TerYang
func:   input the data file address,find the difference between data before and after processing,
        write down the summary report,at the same time,every compared with an unique type list int values,to check
        whatever had changed.
        entry: multipool(),the index of all address source data located and processed data located
"""
import pandas as pd
import numpy as np
import os
from queue import Queue
import random
import multiprocessing as mp
import threading as td
import ctypes

np.set_printoptions(suppress=False,precision=6)

"""
import ctypes
for id in [186, 224, 178]:
    tid = ctypes.CDLL('libc.so.6').syscall(id)  #syscall系统调用
"""
def writeline(file_attr,data):
    with open(file_attr,'a',encoding='utf-8') as f:
        f.writelines(data +'\n')



# 生疏据，第一遍处理，ID_time ,多进程处理
source_attr =["/home/gjj/PycharmProjects/ADA/data/CAN-Instrusion-dataset/",
              "/home/gjj/PycharmProjects/ADA/dealed_data/instrusion_data/",
              "/home/gjj/PycharmProjects/ADA/ID-P/ID_TIME/instrusion_data/",
              "/home/gjj/PycharmProjects/ADA/ID-P/ID_TIME_instrusion_data/"]
dire_attr = "/home/gjj/PycharmProjects/ADA/check_data/round2"

if not os.path.exists(dire_attr):  # 如果保存模型参数的文件夹不存在则创建
    os.makedirs(dire_attr)

attrs = []
for s_attr in source_attr:
    attrs.append(os.listdir(s_attr))


def count_num(q,url,random_num, attr1):
    data = pd.read_csv(url,sep='\s+',delimiter=',', header=None, engine='python', chunksize=5000,dtype=np.str)# ,dtype=np.str
    num = 0

    in_num = []
    for j,o1 in enumerate(data):
        if j ==0:
            print("father pid:{},processing id:{} ,threaing id:{} ".
                  format(os.getppid(),os.getpid(), ctypes.CDLL('libc.so.6').syscall(186)))
        row_size = o1.shape[0]

        if j%100==0 and j !=0:
            print("current:{} at:{}".format(attr1,j))
        num += row_size
        for r_num in random_num:
            qq = [j, r_num, '']
            if r_num < row_size:
                las = np.array(o1.iloc[r_num, :].values,dtype=np.str).tolist()
                for i in las:
                    qq[2] += i
            else:
                qq[2] += 'current total row numbers'+ str(row_size)
            in_num.append(qq)
    q.put(num)
    q.put(in_num)
    print("finished file:{}  at:{}".format(attr1, url))
    print("------")



def multip(url):#(url_r1,url_w1,url_r2,url_w2)

    url_r1 = url[0]
    url_w1 = url[1]
    url_r2 = url[2]
    url_w2 = url[3]

    ran_num = [random.randint(0, 10000) for _ in range(5)]

    q1 = Queue()
    q2 = Queue()
    summ_attr =dire_attr + 'summary.txt'

    try:
        attr1 = url_r1[url_r1.index('t/')+2:url_r1.index('.')]
    except ValueError:
        attr1 = url_r1[url_r1.index('a/') + 2:url_r1.index('.')]

    t1 = td.Thread(target=count_num,args=(q1,url_r1,ran_num,attr1))
    t2 = td.Thread(target=count_num,args=(q2,url_r2,ran_num,attr1))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    nums = [q1.get(),q2.get()]

    if nums[0]==nums[1]:
        a1 = np.array(q1.get()).reshape(-1,3)
        a2 = np.array(q2.get()).reshape(-1,3)
        np.savetxt(url_w1,a1,fmt='%s',delimiter='\t')
        np.savetxt(url_w2,a2,fmt='%s',delimiter='\t')
        # lll1 = pd.DataFrame(q1.get())
        # lll2 = pd.DataFrame(q2.get())

        # lll1.to_csv(url_w1, sep=' ', index=False, header=False, )  # write_urlmode='a'
        # lll2.to_csv(url_w2, sep=' ', index=False, header=False,)  # write_url mode='a'
    else:
        writeline(summ_attr,'{}difference between first file row:{}\t,and second file row:{}'.format(attr1,nums[0], nums[1]))
        writeline(summ_attr,'first at:{}'.format(url_r1))
        writeline(summ_attr,'second at:{}'.format(url_r2))
        writeline(summ_attr, '*************************************************************************************')
        writeline(summ_attr, '\t')


        print('diffrence : first file row\t{}\tsecond file row\t{}'.format(nums[0], nums[1]))
        print("****************************finished:{}**************************************\n".format(attr1))


def multipool(index1,index2):

    write_1_urls = []
    read_1_urls = []
    write_2_urls = []
    read_2_urls = []

    name_queue = Queue()

    dire_attrr = dire_attr + str(index1) + '/'

    """write sumary file"""
    summary_attr = dire_attr +'summary-' + str(index1) + '.txt'

    if not os.path.exists(dire_attrr):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(dire_attrr)

    """找出生数据和第一遍处理数量，第一遍和ID_Time的关系，多进程和ID_time的关系"""
    for attr in attrs[index1]:
        if attr in attrs[index2]:
            write_1_urls.append(dire_attrr + attr[0: attr.index('.')] + "_index{}.txt".format(index1))
            write_2_urls.append(dire_attrr + attr[0: attr.index('.')] + "_index{}.txt".format(index2))
            read_1_urls.append(source_attr[index1] + attr)
            read_2_urls.append(source_attr[index2] + attr)

    pool1 = mp.Pool(processes=len(write_1_urls))#7
    pool1.map(multip,zip(read_1_urls,write_1_urls,read_2_urls,write_2_urls),)

    pool1.close()
    pool1.join()


if __name__ == "__main__":
    multipool(0,1)
    multipool(1,2)
    multipool(2,3)
    print("--all pools had finished!!--")
    # os.system("pstree -p " + str(os.getpid()))
    # #os.getpid()获取当前进程id     获取父进程id




