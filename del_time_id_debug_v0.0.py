import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import threading as td
# from concurrent.futures import ThreadPoolExecutor
import threadpool as tp
import time


# source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion_dataset/origin_data/"
source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion_dataset/origin_data/"
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/'
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/result/RPM_data_cal_time.txt'
dire_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion_dataset/batch_deal_ID_Timediff_v0.3/"


def writelog(content):
    # a = './logs/'
    # a = '/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/secondTry_sameIDsubTime/'
    a = dire_addr +'log/'
    if not os.path.exists(a):
        os.makedirs(a)
    url = a + 'log.txt'
    with open(url, 'a', encoding='utf-8') as f:
        f.writelines('\n'+content + '\n')


def Normalize(str_or_list,flag=None):#data.iloc[,:],传入全部内容，row index，message contents,id，dlc
    a1 = ['a', 'b', 'c', 'd', 'e', 'f']
    lens = len(str_or_list)
    results = []
    """归一化字符数据"""
    np.set_printoptions(suppress=None, precision=3)
    if flag == 'ID':
        np.set_printoptions(suppress=None, precision=3)
        for i, elem in enumerate(str_or_list):
            if i ==0:
                try:
                    results.append(round(np.float64(elem)/7,3))
                except ValueError:
                    print('ID Error at {}'.format(str_or_list))
                continue

            try:
                results.append(round(np.float64(elem)/15,3))
            except ValueError:
                results.append(round(np.float64(10 + a1.index(elem)) / 15,3))
    elif flag == 'C':

        for elem in str_or_list:
            np.set_printoptions(suppress=None, precision=3)
            try:
                results.append(round(np.float64(elem)/15,3))
            except ValueError:
                results.append(round(np.float64(10 + a1.index(elem)) / 15,3))
    return results


def round_str(str_or_list):
    results = []
    for item in str_or_list:
        results.append(round(float(item),4))
    return results

def job(url):
# def job(read_url,write_url,intr_point=None):

    read_url = url[0]
    write_url = url[1]
    data = pd.read_csv(read_url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    # print('data', data.shape)
    data1 = data.copy()

    """扩充data规模"""
    df2 = pd.DataFrame([[5, 6, 5, 6, 5, 6, 5, 6, 5, 6]], index=[0], columns=range(12, 22))
    data1 = data1.join(df2)
    print('reshape before:{},after:{}'.format(data.shape,data1.shape))
    # print('data1',data1.shape)
    # print('data',data.shape)


    num = data.shape[0]
    data.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列
    columns = [i for i in range(data.shape[1])]
    # data.columns = ['Time', 'ID', 'DLC', 0, 1, 2, 3, 4, 5, 6, 7,8]
    data.columns = columns
    # print(data.iloc[:10,:])

    # ll = o1['Time'].groupby(o1['ID'])
    ll = data[0].groupby(data[1])

    names = list(ll.groups.keys())

    print("{} pid processing current: {}, len_name: {},data_shape:{}".
          format(os.getpid(), read_url[-16:], len(names), data.shape))

    writelog("{} pid processing current: {}, len_name: {},data_shape:{}".
             format(os.getpid(), read_url[-16:], len(names), data.shape))

    # print(len(names))
    for name, group in ll:  #per group name是ID名字，group 是该ID包含的index 内容和time 内容
        print("{} pid processing current: {}, name:{}".
          format(os.getpid(), read_url[-16:], name))
        try:
            name = name[1:]
        except IndexError:
            print('file:{},warning at name IndexError:{}'.format(read_url[-16:], name))
        # print(name,'\n',type(group))

        np.set_printoptions(suppress=None, precision=3)
        indexs = group.index.tolist()  # 相同ID的所有全局索引信息
        values = group.values.astype(np.float64).tolist()  # 相同ID的所有全局索引位置的time内容
        # values = group.values.astype(np.str).tolist()  # 相同ID的所有全局索引位置的time内容

        # print(indexs)

        # values = round_str(values)
        """Time"""
        values_ = values.copy()
        # values.insert(0, round(values[0],4))
        values.insert(0, values[0])
        np.set_printoptions(suppress=None, precision=3)
        # data.loc[indexs, 0] = np.array(values_) - np.array(values[:-1])
        group_time = np.array(values_) - np.array(values[:-1])
        group_time = group_time.reshape((-1,1)).round(5)

        """处理message contents归一化"""

        group_id = Normalize(name,'ID')
        # print(group_id,type(group_id),group_id[0],group_id[1],group_id[2])
        # group_id = np.array(group_id).reshape((1,-1))

        # id
        g = np.zeros((len(indexs),3),dtype=np.float64)
        # print(g.shape)
        g[:,0] = g[:,0] + group_id[0]
        g[:, 1] = group_id[1] + g[:, 1]
        g[:,2] = group_id[2] + g[:,2]

        # dlc
        group_dlcs = data.loc[indexs,2].values.astype(np.float32) / 8
        group_dlcs = group_dlcs.reshape((-1,1))

        group_contents = data.loc[indexs,3:].values
        contents = group_contents.copy()

        labels = []
        flag = 0
        np.set_printoptions(suppress=None, precision=3)

        # label and contents
        count = 0
        for rows in group_contents:
            count += 1
            if count%1000==0:
                print("{} pid processing current: {},name: {},index:{}".
                      format(os.getpid(), read_url[-16:], name, count))
            rows = rows.tolist()
            stop_cont = 0
            content = ''
            content_ = []
            for j, row in enumerate(rows):
                if row == 'R':
                    labels.append(1)
                    stop_cont = j
                    break
                elif row == 'T':
                    labels.append(0)
                    stop_cont = j
                    break
                content += row
            stop_cont = 8-stop_cont

            if stop_cont == 8:
                content_ = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            elif stop_cont == 0:
                content_ = Normalize(content,'C')
            else:
                content_ = Normalize(content,'C')
                # print(len(content_))
                for i in range(2*stop_cont):
                    content_.append(0)
                # print('buquan:',len(content_))

            if flag == 0:
                contents = np.array(content_).reshape((1, -1))
                flag = 1
            else:
                # print(contents.shape)
                # print(contents)
                # print()
                contents = np.concatenate((contents, np.array(content_).reshape((1, -1))),axis=0)
        contents = np.concatenate((group_time,g,group_dlcs,contents,np.array(labels).reshape((-1,1))),axis=1)
        data1.loc[indexs,:] = contents
        # print(contents)
        # # columns = [i for i in range(22)]
        # # # data.columns = ['Time', 'ID', 'DLC', 0, 1, 2, 3, 4, 5, 6, 7,8]
        # # data.columns = columns
        # data.iloc[indexs, 1:] = contents
    np.set_printoptions(suppress=None, precision=3)
    data1.to_csv(write_url, sep=' ',index=False, header=False, mode='a', float_format='%.2f')  # write_url

    print('finished:{} file named:{},total numbers:{},{}'.format(os.getpid(),read_url[70:],num,'*'*40))
    try:
        writelog('finished:{} file named:{},total numbers:{},piece numbers:{},{}'.
                 format(os.getpid(),read_url[70:],num,piece_num,'*'*40))

    except NameError:
        writelog('Error at finishe writelog func,is finished')


if __name__ == "__main__":
    # source_attr = r"F:/instrusion_data/"
    # dire_attr = r"F:/ID_TIME_instrusion_data/"
    # source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion dataset/"
    # dire_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/fourthTry_sameIDsubTime/"
    print('program  start at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    print('program  start at:',  time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))

    addrs = os.listdir(source_addr)
    if not os.path.exists(dire_addr):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(dire_addr)
    # pool = mp.Pool(processes=4)
    # notdealed = ['Attack_free_dataset2.txt']
    # source_url = ''
    # dire_url = ''



    # with ThreadPoolExecutor(len(attrs)) as executor:
    #     for addr in attrs:
    #         source_url = source_addr + addr
    #         dire_url = dire_addr + addr
    #         print(source_url)
    #         print(dire_url)
    #         executor.submit(job,args=(source_url, dire_addr,))

    """多线程编程"""
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


    """多进程"""
    # source_url = []
    # dire_url = []
    # interrupt_points = []
    # interr_point = {'RPM_dataset.csv': 404,'gear_dataset.csv': 404,'Fuzzy_dataset.csv': 383}

    source_url = []
    dire_url = []
    interrupt_points = []
    interr_point = {'RPM_dataset.csv': 404,'gear_dataset.csv': 404,'Fuzzy_dataset.csv': 383}

    for addr in addrs:
        source_url.append(source_addr + addr)
        dire_url.append(dire_addr + addr[0: addr.index('.')] + r".txt")
        # dire_url.append(dire_addr + addr)
    print(source_url)
    print(dire_url)
    # exit()
    # job(source_url[0],dire_url[0])#,interrupt_points[1]


    # exit()
    # p1 = mp.Process(target=job,args=(source_url,dire_url),name='p1')
    # p1.start()
    # p1.join()
    # print(source_url)
    # print(dire_url)
    # exit()
    """多进程"""
    pool = mp.Pool(processes=len(source_url))
    pool.map(job, zip(source_url, dire_url),)
    pool.close()
    pool.join()

    """处理Attack_free_dataset2 id只有导致异常问题"""

    # writelog('all processing and program finished at:', 'program run at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    writelog('all processes finished at:{}'.format(time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))))

    print('program  finished at:',  time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))





