import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import threading as td
# from concurrent.futures import ThreadPoolExecutor
import threadpool as tp

def job(url):
# def job(read_url,write_url):
#     print(len(url))
#     print(type(url))
    read_url = url[0]
    write_url = url[1]
    print('read_url',read_url)
    print('write_url',write_url)
    # exit()
    data = pd.read_csv(read_url, sep=None, header=None, dtype=np.str, engine='python', chunksize=10000)  # ,dtype=np.str

    ids_list = []
    larest = {}
    num = 0  # record the number of unique id
    # print(11)
    for j, o1 in enumerate(data):
        # print(o1.loc[36:37,4:])
        # exit()
        print("current: {} at: {}".format(read_url[read_url.index('data/')+5: read_url.index('.')], j))
        # continue
        """dtype: object """
        try:
            o1 = o1.drop(4, axis=1)
        except KeyError or ValueError:
            pass
        o1.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列
        o1.columns = ['Time', 'ID', 'DLC', 0,1,2,3,4,5,6,7]
        # processing_block()
        # print(o1.iloc[36:37,0:].dtypes)
        # exit()
        """df是分块读取的data，j是当前块的顺序,一个名称是name内容是time的groupby数据"""
        ll = o1['Time'].groupby(o1['ID'])
        for name, group in ll:#name是ID名字，group 是该ID包含的index 内容和time 内容
            if len(name)>4:
                name = name[1:-2]
            elif len(name) == 4:
                name = name[1:]
            index_series = group.index.tolist()#相同ID的所有全局索引信息
            values = group.values.tolist()#相同ID的所有全局索引位置的time内容
            # print(index_series,'\n',values)
            # exit()
            """更改ID全局位置的ID信息，缩短ID位数为三位"""
            for index in index_series:
                o1.loc[index, 'ID'] = name

            np.set_printoptions(suppress=True, precision=4)

            """列表操作，更新o1"""
            if j != 0:
                """上一个o1出现过当前相同ID，有多少index，更改多少次，len（values）比len（indexs）长1"""
                if name in larest.keys():#针对上一个o1里留下来的相同ID名称的TIME信息，
                    values.insert(0, larest[name])#该信息插入到当前相同ID的time列表
                    # flag = 1
                    for i in range(len(values) - 1):
                        # print('values index: ', i, '\t', name, 'in range sub: ',
                        #       np.float64(values[i + 1]) - np.float64(values[i]), ' flag: ', flag)
                        o1.loc[index_series[i], 'Time'] = np.float64(values[i + 1]) - np.float64(values[i])

                else:
                    """上一个o1没有出现过当前相同ID，len（values）和len（indexs）一样长"""
                    """#历史没有记录print(name + '\tvalues\t', values)"""
                    if len(values) > 1:
                        for i in range(len(index_series)-1):
                            o1.loc[index_series[i + 1], 'Time'] = np.float64(values[i + 1]) - np.float64(values[i])

                        # print(i, '\t', name, 'in range sub: ', np.float64(values[i + 1]) - np.float64(values[i]))
                    """第一个出现即表示为0"""
                    o1.loc[index_series[0], 'Time'] = 0
            else:
                if len(values) > 1:
                    # print(name +'values\t', values)
                    for i in range(len(index_series)-1):
                        # print(i, '\t', name, 'in range sub: ',np.float64(values[i + 1]) - np.float64(values[i]), ' flag: ', flag)

                        o1.loc[index_series[i + 1], 'Time'] = np.float64(values[i+1]) - np.float64(values[i])
                        """第一个位置"""
                o1.loc[index_series[0], 'Time'] = 0

            # if j == 2:
            #     print(name +' values: ', values, '\tindex\t', index_series)
            """更新字典"""
            larest[name] = np.float64(values[-1])
        o1.to_csv(write_url, sep=' ', index=False, header=False, mode='a', float_format='%.4f')  # write_url


if __name__ == "__main__":
    # source_attr = r"F:/instrusion_data/"
    # dire_attr = r"F:/ID_TIME_instrusion_data/"
    source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion dataset/"
    dire_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/sameIDsubTime/"
    attrs = os.listdir(source_addr)
    # print(attrs)
    # exit()
    if not os.path.exists(dire_addr):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(dire_addr)
    # pool = mp.Pool(processes=4)
    # notdealed = ['Attack_free_dataset2.txt']
    # source_url = ''
    # dire_url = ''

    func = []
    # source_url = []
    # dire_url = []

    # with ThreadPoolExecutor(len(attrs)) as executor:
    #     for addr in attrs:
    #         source_url = source_addr + addr
    #         dire_url = dire_addr + addr
    #         print(source_url)
    #         print(dire_url)
    #         executor.submit(job,args=(source_url, dire_addr,))


    for attr in attrs:
        # if attr in notdealed:
        # source_url.append(source_addr + attr)
        # dire_url.append(dire_addr + attr)#attr[0: attr.index('.')] + r"_ID.txt")
        source_url = []
        source_url.append(source_addr + attr)
        source_url.append(dire_addr + attr)#attr[0: attr.index('.')] + r"_ID.txt")
        # print(source_url)

        # func.append((source_url,None))
        func.append(source_url)
    # print(func)
    # exit()
    """多线程编程"""
    pool = tp.ThreadPool(len(attrs))
    requests = tp.makeRequests(job,func)
    [pool.putRequest(req) for req in requests]
    pool.wait()
    # print(source_url[0],dire_url[0])
    # exit()
    # job(source_url[0],dire_url[0])
    # print(source_url)
    # exit()
    # p1 = mp.Process(target=job,args=(source_url,dire_url,),name='p1')
    # p1.start()
    # p1.join()
    # # print(source_url)
    # # print(dire_url)
    # # exit()

    # pool.map(job, zip(source_url, dire_url),)
    # pool.close()
    # pool.join()

    """处理Attack_free_dataset2 id只有导致异常问题"""
    print("had finished!!")




