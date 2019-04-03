import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import threading as td
# from concurrent.futures import ThreadPoolExecutor
# import threadpool as tp
import time
from decimal import *

source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion_dataset/origin_data/"
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/'
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/result/RPM_data_cal_time.txt'
dire_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion_dataset/batch_deal_ID_Timediff_v0.2/"


def writelog(content):
    # a = './logs/'
    # a = '/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/secondTry_sameIDsubTime/'
    a = dire_addr + 'log/'
    if not os.path.exists(a):
        os.makedirs(a)
    url = a + 'log1.txt'
    with open(url, 'a', encoding='utf-8') as f:
        f.writelines('\n'+content + '\n')


def Normalize(data,index,contents,name,dlc):#data.iloc[,:],传入全部内容，row index，message contents,id，dlc
    a1 = ['a', 'b', 'c', 'd', 'e', 'f']

    """归一化字符数据"""
    np.set_printoptions(precision=3, suppress=True)

    col_index = 1
    contains = [name, dlc, contents]
    for i, contain in enumerate(contains):
        contain = str(contain)
        # print(contain)
        lens = len(contain)
        if i ==0:#name processing
            for _ in range(lens):
                a = 0
                elem = contain[_]
                if _ == 0:
                    # data.at[index,col_index] = float(contain[0])/7
                    a = np.float32(elem)/7#name first bype just hav 3 bits,max is 111,is 7
                else:
                    try:
                        a = np.float32(elem)/15
                    except ValueError:
                        a = np.float32(10 + a1.index(elem)) / 15
                a = round(a, 5)
                data.at[index, col_index] = a
                col_index += 1

        elif i ==1:# dlc processing
            data.at[index, col_index] = round(np.float32(contain)/8,5)
            col_index += 1

        else:
            for _ in contain:
                a = 0
                # print(_)
                try:
                    a = np.float32(_) / 15
                    # one_row.append()
                except ValueError:
                    a = np.float32(10 + a1.index(_)) / 15
                a = round(a, 5)
                data.at[index, col_index] = a
                col_index += 1


# def job(url):
def job(read_url,write_url,intr_point=None):


    data = pd.read_csv(read_url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')

    num = 0  # record the number of unique int
    data.dropna(axis=1, how='all')  # how = ['any','all'] ????
    data.columns = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11]#0:Time,1:ID,2:DLC
    ll = data.groupby(1)
    names = list(ll.groups.keys())
    groups = ll.groups
    # print(len(names))
    print("{} pid processing current: {}, len_name: {},data_shape:{}".
             format(os.getpid(),read_url[-16:],len(names),data.shape))

    writelog("{} pid processing current: {}, len_name: {},data_shape:{}".
             format(os.getpid(),read_url[-16:],len(names),data.shape))
    getcontext().prec = 2

    print('diff names numbers:',len(names))
    writelog('diff names numbers:'.format(len(names)))

    for name, indexs in groups.items():

        # print(type(name))
        # print("name:",name)
        indexs = indexs.tolist()
        name = np.str(name)
        try:
            name = name[1:]
        except IndexError:
            print('file:{},warning at name IndexError:{}'.format(read_url[-16:], name))
        lens = len(indexs)
        print('current at :{},len of indexs:{}'.format(name,lens))
        writelog('current at :{},len of indexs:{}'.format(name,lens))

        """####processing(replace) name for every No.1 index in difference groups or indexs(list) ####"""
        last = data.loc[indexs[0], 0]#index No.1 data Time
        # print('last type:',type(last))
        cout_cont = int(data.loc[indexs[0], 2])

        try:
            data.loc[indexs[0], 1] = name
            data.loc[indexs[0], 0] = 0
        except IndexError or ValueError or TypeError:
            print('file:{},warning at name:{},index:{},at name'.format(read_url[-16:], name, indexs[0]))

        l = 3  # slide index for
        contents = ''#message contents
        try:
            min_ = l
            max_ = cout_cont+l
            if cout_cont:
                for l in range(min_,max_):
                    contents += str(data.loc[indexs[0], l])
                conpletion = 16 - len(contents)
                if conpletion:
                    for _ in range(conpletion):
                        contents += '0'
            else:
                contents += 'ffffffffffffffff'
        except IndexError:
            print('file:{},warning at name:{},index:{},at conpletion'.format(read_url[-16:], name, indexs[0]))

        l = l + 1
        flag = data.loc[indexs[0], l]

        """规格化"""
        Normalize(data,indexs[0],contents,name,cout_cont)

        if flag == 'R':
            data.at[indexs[0],21] = 1
        elif flag == 'T':
            data.at[indexs[0],21] = 0


        # print("indexs:",indexs)
        #
        # print("name:", name)
        # print('DLC:',cout_cont)
        # print('contents:',contents)
        # print('flag:',flag)
        #
        # print('data.iloc[indexs[0], :]:\n',data.iloc[indexs[0], :])
        # print()

        """####processing message contents for every except No.1 index in difference groups or indexs(list) ####"""
        if lens > 1:
            for i in range(0, lens-1):
                if i %1000 == 0:
                    print('current index at:{}'.format(i))
                    writelog('current index at:{}'.format(i))
                cout_cont = int(data.loc[indexs[i + 1], 2])
                l = 3
                try:
                    # diff = Decimal(float(data.loc[indexs[i+1], 0]) - float(last))
                    diff = float(data.loc[indexs[i+1], 0]) - float(last)
                    diff = round(np.float32(diff),5)
                    # print('diff:',diff)
                    last = data.loc[indexs[i+1], 0]
                    # print('diff:',np.float32(diff))
                    data.loc[indexs[i + 1], 0]= diff
                    # print('last:',  Decimal(float(last)))
                except IndexError or ValueError or TypeError:
                    print('file:{},warning at name:{},index:{} at name'.format(read_url[-16:], name, i+1))

                contents = ''
                try:

                    if cout_cont:
                        min_ = l
                        max_ = cout_cont+l
                        for l in range(min_, max_):
                            contents += str(data.loc[indexs[i+1], l])
                        conpletion = 16 - len(contents)
                        if conpletion:
                            for _ in range(conpletion):
                                contents += '0'
                        # data.loc[indexs[i+1], 0] = contents
                    else:
                        # data.loc[indexs[i+1], 0] += 'ffffffffffffffff'
                        contents += 'ffffffffffffffff'
                except IndexError or ValueError or TypeError:
                    print('file:{},warning at name:{},index:{} at conpletion'.format(read_url[-16:], name, i+1))

                l = l + 1
                # print('l:',l)
                flag = data.loc[indexs[i+1], l]

                """规格化"""
                Normalize(data, indexs[i+1], contents, name, cout_cont)

                if flag == 'R':
                    data.at[indexs[i+1], 21] = 1
                elif flag == 'T':
                    data.at[indexs[i+1], 21] = 0

                # print('indexs:',indexs[i + 1])
                # print("indexs:", indexs)
                #
                # print("name:", name)
                # print('cout_cont', cout_cont)
                #
                # print('contents:',contents)
                #
                # print('flag:', flag)
                # print('data.iloc[indexs[i+1],:]:\n',data.iloc[indexs[i+1],:])
                # print()
        # print('end:',name)
        # print()
    # exit()
    data.to_csv(write_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')  # write_url

    print("finished {} pid processing current: {}, len_name: {},data_shape".format(os.getpid(), read_url[-16:], len(names), data.shape))
    try:
        writelog("finished {} pid processing current: {}, len_name: {},data_shape".format(os.getpid(), read_url[-16:], len(names), data.shape))

    except NameError:
        writelog('Error at finished writelog func,is finished')


if __name__ == "__main__":

    addrs = os.listdir(source_addr)
    # print(addrs)
    # exit()
    if not os.path.exists(dire_addr):
        os.makedirs(dire_addr)


    """?????"""
    # func = []
    # for attr in attrs:
    #     # if attr in notdealed:
    #     # source_url.append(source_addr + attr)
    #     # dire_url.append(dire_addr + attr)#attr[0: attr.index('.')] + r"_ID.txt")
    #     source_url = []
    #     source_url.append(source_addr + attr)
    #     source_url.append(dire_addr + attr)#attr[0: attr.index('.')] + r"_ID.txt")
    #     # print(source_url)
    #
    #     # func.append((source_url,None))
    #     func.append(source_url)
    # print(func)
    # exit()
    #
    # pool = tp.ThreadPool(len(attrs))
    # requests = tp.makeRequests(job,func)
    # [pool.putRequest(req) for req in requests]
    # pool.wait()


    """???"""
    source_url = []
    dire_url = []
    interrupt_points = []
    interr_point = {'RPM_dataset.csv': 404,'gear_dataset.csv': 404,'Fuzzy_dataset.csv': 383}

    for addr in addrs:
        source_url.append(source_addr + addr)
        # dire_url.append(dire_addr + addr[0: addr.index('.')] + r"_cal_time_v1.txt")
        dire_url.append(dire_addr + addr)

    job(source_url[1],dire_url[1])#,interrupt_points[1]

    # p1 = mp.Process(target=job,args=(source_url,dire_url),name='p1')
    # p1.start()
    # p1.join()

    """???"""
    # writelog('all processes finished at:{}'.format(time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))))
    # print('program run at:', 'program run at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    # pool = mp.Pool(processes=len(source_url))
    # pool.map(job, zip(source_url, dire_url,interrupt_points),)
    # pool.close()
    # pool.join()

    """??Attack_free_dataset2 id????????"""

    # writelog('all processing and program finished at:', 'program run at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    writelog('all processes finished at:{}'.format(time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))))

    print('program  finished at:',  time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))




