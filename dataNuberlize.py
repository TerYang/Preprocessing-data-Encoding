import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import threading as td
# from concurrent.futures import ThreadPoolExecutor
# import threadpool as tp
import time
# from sklearn.preprocessing import MinMaxScaler
a1 = ['a', 'b', 'c', 'd', 'e', 'f']

# source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/origin-data/"
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/origin-data/'
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/'
# source_addr = '/home/gjj/PycharmProjects/ADA/raw_data/test/result/RPM_data_cal_time.txt'
# dire_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/"


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
    # print(content)
    with open(url, 'a', encoding='utf-8') as f:
        f.writelines('\n'+content + '\n')


def Normalize(str_or_list,flag=None):#data.iloc[,:],传入全部内容，row index，message contents,id，dlc
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

def chartoNumber(str):
    result = []
    for i in str:
        if i>= "a" and i<= "f":
            try:
                result.append(10 + a1.index(i))
            except ValueError:
                print("处理id 字符转整形出错,{}".format(str))
        else:
            result.append(int(i))
    return result

# def job(url):
def job(read_url,write_url):

    # readurl1 = os.path.splitext(read_urls[0]) #'.csv'分离格式与其他
    # print(readurl1)
    # print(os.path.basename(read_urls[0]))#DoS_dataset.csv，获取文件名
    # print( os.path.dirname(read_urls[0]) )#获取路径
    # #/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset
    # # print(os.path.splitunc(read_urls[0]) )
    # print(os.path.split(read_urls[0]) )#获取路径与文件名

    # read_url = url[0]
    # write_url = url[1]
    filename = os.path.basename(read_url)

    log_url = os.path.join(dire_addr,'logs',os.path.splitext(filename)[0]+'.txt')
    data = pd.read_csv(read_url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data1 = data.copy()
    # print(data.loc[:10,:])
    """扩充data规模"""
    # df2 = pd.DataFrame([[5, 6, 5, 6, 5, 6, 5, 6, 5, 6]], index=[0], columns=range(12, 22))
    # df2 = pd.DataFrame([[5, 6,]], index=[0], columns=range(5, 7))
    if data.shape[1]== 4:
        df2 = pd.DataFrame([[5, 6,]], index=[0], columns=range(4,6))
        data1 = data1.join(df2)
    elif data.shape[1] == 5:
        df2 = pd.DataFrame([[5]], index=[0], columns=range(5, 6))
        data1 = data1.join(df2)
    # writelog('{} reshape before:{},after:{}'.format(filename,data.shape,data1.shape),log_url)
    # print(data1.loc[:10,:])
    print('data1',data1.shape,data1.columns,end="  *##*  ")
    print('data',data.shape,data.columns)
    # print(data.columns)
    # print(data1.columns)
    # exit()

    num = data.shape[0]
    data.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列
    columns = [i for i in range(data.shape[1])]
    data.columns = columns
    # ll = o1['Time'].groupby(o1['ID'])
    # 获取时间戳，并以ID 为进行分组
    ll = data[0].groupby(data[1])

    names = list(ll.groups.keys())
    pid = os.getpid()
    piece_num = len(names)
    print("{} pid processing current: {}, len_name: {},data_shape:{}".
          format(pid, filename, len(names), data.shape))

    writelog("{} pid processing {}, diff names size: {},file size:{}".
             format(pid, filename, piece_num, data.shape),log_url)

    for name, group in ll:
        #per group name是ID名字，每个 ID 的group 包含索引和时间戳的值
        if name == 'None':
            writelog('{} {} NoneError'.format(filename, name), log_url)
            continue
        elif len(name) > 3:
            name = name[:4]
        np.set_printoptions(precision=5)
        indexs = group.index.tolist()  # 相同ID的所有全局索引信息

        writelog("{} pid processing file: {}, ID:{}, group size:{}".format(pid, filename, name,len(indexs)),log_url)

        try:
            values = group.values.astype(np.float64).tolist()  # 相同ID的所有全局索引位置的time内容
        except ValueError:
            writelog('{} data type error at{}'.format(filename, name),log_url)
        values_ = values.copy()
        values.insert(0, values[0])
        np.set_printoptions(suppress=True,precision=5)
        group_time = np.subtract(group.values.astype(np.float64).reshape((-1,1)),np.array(values[:-1]).reshape((-1,1)))
        group_time = group_time.reshape((-1,1))#.round(5)
        group_time = np.multiply(group_time,1000).round(2)
        """处理message contents归一化"""
        # group_id = Normalize(name,'ID')
        """将ID字符串转为数值"""
        group_id = chartoNumber(name)
        #### id ###################################################
        g = np.zeros((len(indexs),4),dtype=np.float64)
        # print(g.shape)
        g[:,0] = g[:,0] + group_id[0]
        g[:, 1] = group_id[1] + g[:, 1]
        g[:,2] = group_id[2] + g[:,2]
        g[:,3] = group_id[3] + g[:,3]

        # dlc ####################################################
        # group_dlcs = data.loc[indexs,2].values.astype(np.float32) / 8
        group_dlcs = data.loc[indexs,2].values.astype(np.int32)
        group_dlcs = group_dlcs.reshape((-1,1))

        np.set_printoptions(precision=5)

        """label and contents"""
        contents = np.concatenate((group_time,g,group_dlcs),axis=1)
        # print(contents.shape)
        # print(data1.loc[indexs,:].shape)
        data1.loc[indexs,:] = np.around(contents, decimals=5)

    np.set_printoptions(suppress=True,precision=5)

    ############# write file ############################################
    data1.to_csv(write_url, sep=' ',index=False, header=False, mode='w', float_format='%.2f')  # write_url

    try:
        writelog('finished:{} file named:{},total numbers:{},piece numbers:{},{}'.
                 format(pid,filename,num,piece_num,'*'*40),log_url)
    except NameError:
        writelog('Error at finishe writelog func,is finished',log_url)
    writelog('{} finished at:{}'.format(filename,time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))),log_url)
    print('finished:{} file named:{},total numbers:{},piece numbers:{},{}'.
                 format(pid,filename,num,piece_num,'*'*40),log_url)

if __name__ == "__main__":
    # source_attr = r"F:/instrusion_data/"
    # dire_attr = r"F:/ID_TIME_instrusion_data/"
    # source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-instrusion dataset/"
    # dire_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/car-hacking-instrusion/fourthTry_sameIDsubTime/"
    source_addr = "F:\Yangyuanda\ADA\dealed_data\instrusion_data"
    dire_addr = "F:\Yangyuanda\ADA\dealed_data\新数据1025"
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
    # interr_point = {'RPM_dataset.csv': 404,'gear_dataset.csv': 404,'Fuzzy_dataset.csv': 383}
    source_urls = [os.path.join(source_addr,addr) for addr in addrs]
    dire_urls = [os.path.join(dire_addr,'result',os.path.splitext(addr)[0]+'.txt') for addr in addrs]
    # print('dire_urls:\n',dire_urls)
    # print('source_urls:\n',source_urls)
    # exit()
    if not os.path.exists(os.path.dirname(dire_urls[0])):
        os.makedirs(os.path.dirname(dire_urls[0]))

    # job(source_urls[0],dire_urls[0])
    # exit()
    for url1,url2 in list(zip(source_urls[3:],dire_urls[3:])):
        job(url1,url2)#,interrupt_points[1]


    # exit()
    # p1 = mp.Process(target=job,args=(source_url,dire_url),name='p1')
    # p1.start()
    # p1.join()
    # print(source_url)
    # print(dire_url)
    # exit()
    """多进程"""
    # pool = mp.Pool(processes=len(source_urls))
    # pool.map(job, zip(source_urls, dire_urls),)
    # pool.close()
    # pool.join()

    """处理Attack_free_dataset2 id只有导致异常问题"""

    # writelog('all processing and program finished at:', 'program run at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    # writelog('all processes finished at:{}'.format(time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))))

    print('program  finished at:',  time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))





