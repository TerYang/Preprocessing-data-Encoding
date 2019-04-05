import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import threading as td
# from concurrent.futures import ThreadPoolExecutor
import threadpool as tp
import time
from sklearn.preprocessing import MinMaxScaler


source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/origin-data/"
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/origin-data/'
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/'
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/result/RPM_data_cal_time.txt'
dire_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/"


def writelog(content,url):
    # a = './logs/'
    # a = '/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/secondTry_sameIDsubTime/'
    # a = dire_addr +'log/'
    # if not os.path.exists(a):
    url_path = os.path.dirname(url)
    # print('os.path.dirname:\n',url_path)

    if not os.path.exists(url_path):
        # os.makedirs(a)
        os.makedirs(url_path)
    # url = a + 'log.txt'
    print(content)
    with open(url, 'a', encoding='utf-8') as f:
        f.writelines('\n'+content + '\n')


def Normalize(str_or_list,flag=None):#data.iloc[,:],传入全部内容，row index，message contents,id，dlc
    a1 = ['a', 'b', 'c', 'd', 'e', 'f']
    lens = len(str_or_list)
    results = []
    """归一化字符数据"""
    np.set_printoptions(precision=3)
    if flag == 'ID':
        np.set_printoptions(precision=3)
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
            np.set_printoptions(precision=3)
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
# def job(read_url,write_url):

    # os

    # readurl1 = os.path.splitext(read_urls[0]) #'.csv'分离格式与其他
    # print(readurl1)
    # print(os.path.basename(read_urls[0]))#DoS_dataset.csv，获取文件名
    # print( os.path.dirname(read_urls[0]) )#获取路径
    # #/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset
    #
    # # print(os.path.splitunc(read_urls[0]) )
    #
    # print(os.path.split(read_urls[0]) )#获取路径与文件名

    read_url = url[0]
    write_url = url[1]
    filename = os.path.basename(read_url)

    log_url = os.path.join(dire_addr,'logs',os.path.splitext(filename)[0]+'.txt')
    # writelog('{} reshape before'.format(filename,), log_url)
    # print('filename:\n',filename)
    # print('log_url:\n',log_url)
    # exit()

    data = pd.read_csv(read_url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    # print('data', data.shape)
    data1 = data.copy()

    """扩充data规模"""
    df2 = pd.DataFrame([[5, 6, 5, 6, 5, 6, 5, 6, 5, 6]], index=[0], columns=range(12, 22))
    data1 = data1.join(df2)
    writelog('{} reshape before:{},after:{}'.format(filename,data.shape,data1.shape),log_url)

    # exit()

    # print('data1',data1.shape)
    # print('data',data.shape)

    # print(data.columns)
    # print(data1.columns)
    # exit()

    num = data.shape[0]
    data.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列
    columns = [i for i in range(data.shape[1])]
    # data.columns = ['Time', 'ID', 'DLC', 0, 1, 2, 3, 4, 5, 6, 7,8]
    data.columns = columns
    # print(data.iloc[:10,:])

    # ll = o1['Time'].groupby(o1['ID'])
    ll = data[0].groupby(data[1])

    names = list(ll.groups.keys())
    pid = os.getpid()

    # print("{} pid processing current: {}, len_name: {},data_shape:{}".
    #       format(pid, filename, len(names), data.shape))

    writelog("{} pid processing {}, diff names size: {},file size:{}".
             format(pid, filename, len(names), data.shape),log_url)

    # print(len(names))
    for name, group in ll:  #per group name是ID名字，group 是该ID包含的index 内容和time 内容

        writelog('{} {} before truncate'.format(filename, name), log_url)
        fl_ag = 0
        if name == 'None':
            writelog('{} {} NoneError'.format(filename, name), log_url)
            continue

        elif len(name) > 3:
            fl_ag = 1
            if name[0] == '0':
                try:
                    name = name[1:4]
                except IndexError:
                    writelog('{} {} IndexError:'.format(filename, name),log_url)
            else:
                try:
                    name = name[:3]
                except IndexError:
                    writelog('{} {} IndexError:'.format(filename, name),log_url)
        elif len(name)<3:
            fl_ag = 1
            name = name+'0'
        if fl_ag:
            writelog('{} {} truncated'.format(filename, name), log_url)

        np.set_printoptions(precision=3)
        indexs = group.index.tolist()  # 相同ID的所有全局索引信息

        writelog("{} pid processing file: {}, ID:{}, group size:{}".format(pid, filename, name,len(indexs)),log_url)

        try:
            values = group.values.astype(np.float64).tolist()  # 相同ID的所有全局索引位置的time内容
        except ValueError:
            writelog('{} data type error at{}'.format(filename, name),log_url)
        # values = group.values.astype(np.str).tolist()  # 相同ID的所有全局索引位置的time内容

        # print(indexs)

        # values = round_str(values)

        ##### Time ##############################################
        values_ = values.copy()
        # values.insert(0, round(values[0],4))
        values.insert(0, values[0])
        np.set_printoptions(suppress=True,precision=3)
        # data.loc[indexs, 0] = np.array(values_) - np.array(values[:-1])
        # group_time = np.array(values_) - np.array(values[:-1])
        # np.savetxt(log_url,group.values.astype(np.float64),delimiter=',',fmt='%.5f')
        # np.savetxt('/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/origin-data/log.txt',np.array(values[:-1]),delimiter=',')
        # print(values_[40:60])
        # print(values[40:60])
        group_time = np.subtract(group.values.astype(np.float64).reshape((-1,1)),np.array(values[:-1]).reshape((-1,1)))
        group_time = group_time.reshape((-1,1)).round(5)
        # print(group_time)
        # print(np.max(group_time))
        # exit()

        """处理message contents归一化"""

        group_id = Normalize(name,'ID')
        # print(group_id,type(group_id),group_id[0],group_id[1],group_id[2])
        # group_id = np.array(group_id).reshape((1,-1))

        #### id ###################################################
        g = np.zeros((len(indexs),3),dtype=np.float64)
        # print(g.shape)
        g[:,0] = g[:,0] + group_id[0]
        g[:, 1] = group_id[1] + g[:, 1]
        g[:,2] = group_id[2] + g[:,2]

        # dlc ####################################################
        group_dlcs = data.loc[indexs,2].values.astype(np.float32) / 8
        group_dlcs = group_dlcs.reshape((-1,1))

        group_contents = data.loc[indexs,3:].values
        contents = group_contents.copy()

        labels = []
        flag = 0
        np.set_printoptions(precision=3)

        # label and contents
        count = 0
        for rows in group_contents:
            count += 1
            if count%100000==0:
                writelog("{} pid processing current: {},name: {},index:{}".
                      format(pid, filename, name, count),log_url)
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

        # print(contents[:4,:4],contents.shape)
        # print(data1.loc[indexs,:].shape)
        data1.loc[indexs,:] = np.around(contents, decimals=4)
        # print(contents)
        # # columns = [i for i in range(22)]
        # # # data.columns = ['Time', 'ID', 'DLC', 0, 1, 2, 3, 4, 5, 6, 7,8]
        # # data.columns = columns
        # data.iloc[indexs, 1:] = contents
    np.set_printoptions(suppress=True,precision=3)

    ############   scaler time to [0,1]############################
    scaler = MinMaxScaler(feature_range=(0, 1))
    scale_a = scaler.fit_transform(np.array(data.loc[:, 0].values.astype(np.float64)).reshape(-1, 1))
    scale_a = np.around(scale_a, decimals=3)
    writelog('finished {} {},scaler sizes {},ndim {}'.format(pid, filename, scale_a.shape,scale_a.ndim), log_url)
    # print(scale_a.shape,scale_a.ndim)

    try:
        data1.loc[:,0] = scale_a
        writelog('merge success',log_url)
    except IndexError or ValueError:
        writelog('merge error',log_url)

    ############# write file ############################################
    data1.to_csv(write_url, sep=' ',index=False, header=False, mode='a', float_format='%.2f')  # write_url

    writelog('finished:{} file named:{},total numbers:{},{}'.format(pid,filename,num,'*'*40),log_url)
    try:
        writelog('finished:{} file named:{},total numbers:{},piece numbers:{},{}'.
                 format(pid,filename,num,piece_num,'*'*40),log_url)

    except NameError:
        writelog('Error at finishe writelog func,is finished',log_url)
    writelog('{} finished at:{}'.format(filename,time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))),log_url)


if __name__ == "__main__":
    # source_attr = r"F:/instrusion_data/"
    # dire_attr = r"F:/ID_TIME_instrusion_data/"
    # source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion dataset/"
    # dire_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/fourthTry_sameIDsubTime/"
    print('program  start at:',  time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))

    addrs = os.listdir(source_addr)

    # if not os.path.exists(dire_addr):  # 如果保存模型参数的文件夹不存在则创建
    #     os.makedirs(dire_addr)
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
    source_urls = [os.path.join(source_addr,addr) for addr in addrs]
    dire_urls = [os.path.join(dire_addr,'result',os.path.splitext(addr)[0]+'.txt') for addr in addrs]
    # print('dire_urls:\n',dire_urls)
    # print('source_urls:\n',source_urls)
    # exit()
    # exit()
    if not os.path.exists(os.path.dirname(dire_urls[0])):
        os.makedirs(os.path.dirname(dire_urls[0]))

    # exit()
    # job(source_urls[1],dire_urls[1])#,interrupt_points[1]


    # exit()
    # p1 = mp.Process(target=job,args=(source_url,dire_url),name='p1')
    # p1.start()
    # p1.join()
    # print(source_url)
    # print(dire_url)
    # exit()
    """多进程"""
    pool = mp.Pool(processes=len(source_urls))
    pool.map(job, zip(source_urls, dire_urls),)
    pool.close()
    pool.join()

    """处理Attack_free_dataset2 id只有导致异常问题"""

    # writelog('all processing and program finished at:', 'program run at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    # writelog('all processes finished at:{}'.format(time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))))

    print('program  finished at:',  time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))





