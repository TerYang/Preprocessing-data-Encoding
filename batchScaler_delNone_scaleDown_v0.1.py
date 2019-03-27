import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import threading as td
import multiprocessing as mp
from queue import Queue
import time

np.set_printoptions(suppress=False, precision=4)
# source_addr ="/home/gjj/PycharmProjects/ADA/ID-P/"


def culcalative_scaler(o1, min_global, max_global, feature_range=(0, 1)):
    """a 是Series"""
    # a = np.array(np.random.normal(0, 1, 10))
    # min_global = np.min(a)
    # max_global = np.max(a)
    # func_std = np.std(a)
    r_min = feature_range[0]
    r_max = feature_range[1]
    for i, elem in enumerate(o1):
        # std_a = (a - min_global) / (max_global - min_global)
        o1.at[i, 0] = ((elem - min_global) / (max_global - min_global)) * (r_max - r_min) + r_min


def batch(np_or_series):
    """(value - mean )/rsqrt(variance) 数值减去均值之差除标准差,归一化后有负值"""
    # mean_value = np.mean(np_or_series)
    # std_value = np.std(np_or_series)
    # np_or_series = (np_or_series - mean_value)/std_value
    # print(np_or_series)
    """归一化在一定范围内"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    print(np_or_series.shape)
    np_or_series = scaler.fit_transform(np_or_series.reshape(-1, 1))
    return np.around(np_or_series, decimals=2)  # 返回精度四舍五入为小数点后两位


def del_None(o1,name):
    for i in o1.index:
        completion_num = 16 - len(o1.at[i, 3])
        if o1.at[i, 3] == 'None':
            # print("is None")
            o1.at[i, 3] = 'ffffffffffffffff'  # 16个1
        elif completion_num:
            for l in range(completion_num):
                o1.at[i, 3] += '0'  # 后面补全0
        if i % 100000 == 0:
            print("scaler:{},at:{}".format(name, i))
    return o1


def findGlobalMaxMin(piece, q):  # piece is data read from file
    piece_min = 0.0
    piece_max = 0.0
    max_value_index = 0
    num = 0
    for i, elem in enumerate(piece):
        # if i%10==0:
        #     print("at{}\n".format(i))
        # print(type(elem.iloc[:7,0]))
        e_max = elem.iloc[:, 0].max()
        e_min = elem.iloc[:, 0].min()
        if i == 0:
            piece_min = e_min
            piece_max = e_max

            continue
        if piece_min > e_min:
            piece_min = e_min
        if piece_max < e_max:
            list_elem = np.array(elem.iloc[:, 0].values).tolist()
            index_max = list_elem.index(e_max)
            max_value_index = i * 10000 + index_max  # 最大数的位置
            piece_max = e_max
            # print("at{}, max:{},list index:{},global:{}".format(i,e_max,index_max,max_value_index))
        # num += elem.shape[0]
        # max_value = o1_max.at[o1_max.index[0]]
    q.put(piece_max)
    q.put(piece_min)
    q.put(max_value_index)


"""trash codes
        time11 = time.time()
        print(type(scale_a), scale_a.shape)
        print("scale_time:\n",time11-time0)
        print(data.loc[20:50,0])
        # exit()
        scaler手工计算方法
        time1 = time.time()
        data = pd.read_csv(read_url, sep=None, header=None, engine='python', chunksize=10000)
        shape (100, 4),Index([0, 1, 2, 3], dtype='int64')
        dtypes: 1     object
                2      int64
                3     object
                dtype: object
"""
"""
        # data = pd.read_csv(read_url, sep=None, header=None, engine='python')
        # data = pd.read_csv(read_url,sep='\s+',delimiter=',', header=None, engine='python', chunksize=50,dtype=np.str)# shape(50, 1)
        name = addr[:addr.index('.')]
        t1 = td.Thread(target=findGlobalMaxMin,args=(data,q,),name=name)
        t1.start()
        t1.join()
        t_max = q.get()
        t_min = q.get()

        data = pd.read_csv(read_url, sep=None, header=None, engine='python', chunksize=10000)
        # print(data[2].loc[20:50, 0])
        # exit()
        for j, o1 in enumerate(data):
            if j%10==0:
                print(j)
            np.set_printoptions(suppress=True,precision=2)
            culcalative_scaler(o1,t_min,t_max)
            if j == 0:
                print(o1.loc[20:50, 0])
                exit()
        time3 = time.time()
        print("time1:",time3-time1)

"""


def one_processing(read_url, write_url):
    name = read_url[read_url.index('data/') + 5:read_url.index('.')]
    a1 = ['a', 'b', 'c', 'd', 'e', 'f']

    """sklearn.preprocessing import MinMaxScaler的方法"""
    data = pd.read_csv(read_url, sep=None, header=None, engine='python')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scale_a = scaler.fit_transform(np.array(data.loc[:, 0].values).reshape(-1, 1))
    scale_a = np.around(scale_a, decimals=3)
    print("{} finish scaler".format(name))
    # print("scale_a.shape:",scale_a.shape)
    # exit()
    """补全数据,归一化字符数据"""
    """补全数据"""
    bb = 0
    data = del_None(data,name)
    print("{} finish del None".format(name))
    np.set_printoptions(precision=4, suppress=None, )
    # print("data.loc[:5,:]:\n",data.loc[:5,:])
    # print()
    count = 0
    for i in data.index:  # i is row index,col is columns index
        # count = 1
        one_row = [scale_a[i, 0]]
        for col in range(1, 4, 1):  # added other data like DLC content

            if col == 2:  # DLC DLC max =8
                a = float(data.at[i, col]) / 8
                one_row.append(a)

            else:
                for elem in data.at[i, col]:  # max = 15
                    try:
                        a = float(elem) / 15
                        # one_row.append()
                    except ValueError:
                        a = float(10 + a1.index(elem)) / 15
                    one_row.append(a)

        if count == 0:
            bb = np.array(list(one_row), dtype=np.float32).reshape(1, -1)
            count += 1
            continue
        bb = np.concatenate((bb, np.array(list(one_row), dtype=np.float32).reshape(1, -1)), axis=0)
        if i % 100000 == 0:
            print("current:{},at:{}".format(name, i))
    bb = pd.DataFrame(bb, dtype=np.float32)
    bb.to_csv(write_url, sep=' ', index=False, header=False, mode='r', float_format='%.3f')
    # # np.savetxt(write_url,bb,fmt='%.4f',delimiter='\t',newline='\n',)
    # np.loadtxt(write_url,bb)

    print("finish:{}".format(name))


if __name__ == "__main__":
    """1.求出时间归一值，2补全content内容，2第二列到第四列内容进行归一化"""
    source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/ID_TIME_instrusion_data/"
    # source_addr ="F:/ID_TIME/instrusion_data/"
    dire_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/"

    addrs = os.listdir(source_addr)

    # print(addrs)
    if not os.path.exists(dire_addr):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(dire_addr)

    source_url = []
    dire_url = []
    had_done = ['Attack_free_dataset2_ID.txt']

    for addr in addrs:
        if addr not in had_done:
            continue
        else:
            source_url.append(source_addr + addr)
            dire_url.append(dire_addr + addr[0: addr.index('.')] + r"_Normalize.txt")

    p1 = mp.Process(target=one_processing, args=(source_url[0], dire_url[0],))
    p1.start()
    p1.join()
    print("finished!!!")
