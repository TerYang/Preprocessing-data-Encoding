import pandas as pd
import numpy as np
import os
import multiprocessing as mp


# def job(url):
def job(read_url,write_url):
    # read_url = url[0]
    # write_url = url[1]
    data = pd.read_csv(read_url, sep=None, header=None, dtype=np.str, engine='python', chunksize=10000)  # ,dtype=np.str

    ids_list = []
    larest = {}
    num = 0  # record the number of unique id

    for j, o1 in enumerate(data):
        # print(o1.loc[:10,:])
        # exit()
        print("current: {} at: {}".format(read_url[read_url.index('data/')+5: read_url.index('.')], j))
        # continue
        """dtype: object """
        try:
            o1 = o1.drop(4, axis=1)
        except KeyError or ValueError:
            pass
        o1.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列
        o1.columns = ['Time', 'ID', 'DLC', 'Comment']
        # processing_block()

        """df是分块读取的data，j是当前块的顺序,一个名称是name内容是time的groupby数据"""
        ll = o1['Time'].groupby(o1['ID'])
        for name, group in ll:
            if len(name)>4:
                name = name[1:-2]
            index_series = group.index.tolist()
            values = group.values.tolist()

            """处理ID"""
            for index in index_series:
                o1.loc[index, 'ID'] = name

            np.set_printoptions(suppress=True, precision=6)
            """列表操作，更新o1"""
            if j != 0:
                if name in larest.keys():#历史有过记录
                    values.insert(0, larest[name])
                    # flag = 1
                    for i in range(len(values) - 1):
                        # print('values index: ', i, '\t', name, 'in range sub: ',
                        #       np.float64(values[i + 1]) - np.float64(values[i]), ' flag: ', flag)
                        o1.loc[index_series[i], 'Time'] = np.float64(values[i + 1]) - np.float64(values[i])

                else:
                    """#历史没有记录print(name + '\tvalues\t', values)"""
                    if len(values) > 1:
                        for i in range(len(index_series)-1):
                            o1.loc[index_series[i + 1], 'Time'] = np.float64(values[i + 1]) - np.float64(values[i])

                        # print(i, '\t', name, 'in range sub: ', np.float64(values[i + 1]) - np.float64(values[i]))
                            """第一个位置"""
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
        o1.to_csv(write_url, sep=' ', index=False, header=False, mode='a', float_format='%.6f')  # write_url


if __name__ == "__main__":
    # source_attr = r"F:/instrusion_data/"
    # dire_attr = r"F:/ID_TIME_instrusion_data/"
    source_attr = '/home/gjj/PycharmProjects/ADA/dealed_data/instrusion_data/'
    dire_attr = '/home/gjj/PycharmProjects/ADA/ID-TIME_data/ID_TIME_instrusion_data/'
    attrs = os.listdir(source_attr)
    # print(attrs)
    # exit()
    if not os.path.exists(dire_attr):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(dire_attr)
    # pool = mp.Pool(processes=4)
    notdealed = ['Attack_free_dataset2.txt']
    source_url = ''
    dire_url = ''
    for attr in attrs:
        if attr in notdealed:
            source_url = source_attr + attr
            dire_url=dire_attr + attr[0: attr.index('.')] + r"_ID.txt"
    # print(source_url)
    # exit()
    p1 = mp.Process(target=job,args=(source_url,dire_url,),name='p1')
    p1.start()
    p1.join()
    # # print(source_url)
    # # print(dire_url)
    # # exit()

    # pool.map(job, zip(source_url, dire_url),)
    # pool.close()
    # pool.join()

    """处理Attack_free_dataset2 id只有导致异常问题"""
    print("had finished!!")




