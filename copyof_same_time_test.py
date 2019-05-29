# -*- coding: utf-8 -*-
# @Time   : 19-5-29 下午9:21
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================

import pandas as pd
import numpy as np
# s = '/home/gjj/PycharmProjects/ADA/raw_data/CAN-Instrusion-dataset/Attack_free_dataset.txt'
# m = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/1/data/Attack_free_dataset.pkl'
# ind_url = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/2/logs/errorindexs.pkl'
#
# indexs = pd.read_pickle(ind_url,compression='zip')
# indexs = indexs.values.squeeze().tolist()[:10]
# print(indexs)
#
# # dats = pd.read_csv(s, sep=None, header=None, dtype=np.str, engine='python', encoding='utf-8')
# datm = pd.read_pickle(m,compression='zip')
#
# # print('source raw data:\n')
# # print(dats.iloc[indexs,:])
#
# c1 = [10, 19, 28,32, 33, 34 , 35, 36, 37, 38, 39]
# c2 = [9, 18, 27, 31, 32, 33 , 34, 35, 36, 37, 38]
# # print(dats.iloc[indexs,c])
# # print('fisrt 20')
# # print(dats.iloc[:20,c2])
# # print()
# # print(dats.iloc[:20,c1])
#
# print('indexs')
# print(datm.iloc[indexs,c2])
# print()
# print(datm.iloc[indexs,c1])
#
# '[22827, 22828, 22829, 22830, 22831, 22832, 22833, 22834, 22835, 22836]'
# # print('meddile data:\n')
# # print(datm.iloc[indexs,:])
#
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import threading as td
# from concurrent.futures import ThreadPoolExecutor
import threadpool as tp
import time
from sklearn.preprocessing import MinMaxScaler
import json
import re
import math
import operator

d = lambda x, y: x if operator.ge(x, y) else y
# f = lambda x: int(x, 16)


def writelog(content,filname=None):
    path = os.path.join('./','logs')
    # logs 目录下,记录所有处理的信息
    try:
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
    except FileExistsError:
        pass

    if filname:
        url_path = os.path.join(path, '{}.txt'.format(filname))
        with open(url_path, 'a', encoding='utf-8') as f:
            if content.__class__ == str:
                f.writelines('\n'+content + '\n')
            else:
                f.writelines('\n')
                f.write(json.dumps(content))
                f.writelines('\n')
    else:
        print(content)


def raw_to_hackingdata(url):
    """
    func:输入,hacking dataset原始数据集,ID为四位的内容.
    计算相邻的相同ID时间间隔,将16进制转化为10进制,补全内容,没有归一化.
    :param url:
    :return:
    """

    pid = os.getpid()
    read_url = url[0]
    write_url = url[1]
    filename = os.path.basename(read_url)
    filename = os.path.splitext(filename)[0]
    # print(filename)
    data = pd.read_pickle(read_url, compression='zip')
    columns = [i for i in range(data.shape[1])]
    # data.columns = ['Time', 'ID', 'DLC', 0, 1, 2, 3, 4, 5, 6, 7,8]
    data.columns = columns
    # print(data.loc[:10,:])
    # count = 0
    # bad = []
    # for i,dat in enumerate(data[1].values.tolist()):
    #     if dat == 'nan':
    #         bad.append(i)
    #         count +=1
    # writelog('id is nan lengh:%d:'%count,filename)
    # bad = pd.DataFrame(bad)
    #
    # bad.to_pickle(os.path.join('./','logs','errorindexs.pkl'),compression='zip')
    # exit()
    if data.shape[1] > 11:
        print(filename, data.shape)
        print(data.loc[:10, :])
        return
    data1 = data.copy()
    writelog('{} pid processing {}'.format(pid, filename))
    """扩充data规模"""
    df2 = pd.DataFrame([[i for i in range(10)]], index=[0], columns=range(11, 21))
    data1 = pd.concat([data1, df2], axis=1)
    # 设置目标数据承载空间
    writelog('{} reshape before:{},after:{}'.format(filename, data.shape, data1.shape), filename)
    print('{} reshape before:{},after:{}'.format(filename, data.shape, data1.shape))
    num = data.shape[0]
    data.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列

    # ll = o1['Time'].groupby(o1['ID'])
    ll = data[0].groupby(data[1])

    names = list(ll.groups.keys())
    writelog("diff names: {},file size:{}".
             format(len(names), data.shape), filename)

    for name, group in ll:
        # per group name是ID名字，group是该ID包含的index 信息和time

        nam = name
        badname = []
        fl_ag = 0
        if name == 'None':
            writelog('{} None Error,id is:{}'.format(filename, name), filename)
            fl_ag = 1

        elif len(name) == 4:

            if name[0] == '0':
                try:
                    name = name[1:4]
                    # writelog('ID:{} truncated as {}'.format(nam, name), filename)
                except IndexError:
                    writelog('{} IndexError: {}'.format(filename, name), filename)
                    return
            else:
                badname.append(name)
                fl_ag = 1
        else:
            fl_ag = 1
            badname.append(name)
            # name = name+'0'
            writelog('{} special error ID:{}'.format(filename, name), filename)

        if fl_ag:
            writelog('error ID has {}'.format(len(badname)), filename)
            badname = np.array(badname).reshape((-1,10))
            u_bad = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/2/logs/badid.txt'
            np.savetxt(u_bad,badname,fmt='%d',delimiter=' ',encoding='utf-8')
            return

        # ###############################   处理name
        np.set_printoptions(precision=3)
        indexs = group.index.tolist()  # 相同ID的所有全局索引信息

        writelog("ID:{} to {}, group size:{}".format(nam, name, len(indexs)), filename)

        try:
            values = group.values.astype(np.float64).tolist()  # 相同ID的所有全局索引位置的time内容
        except ValueError:
            writelog('{} data type error at{}'.format(filename, name), filename)
        # values = group.values.astype(np.str).tolist()  # 相同ID的所有全局索引位置的time内容

        ##### Time ##############################################
        values.insert(0, values[0])
        np.set_printoptions(suppress=True, precision=5)

        # subtraction of adjacent same ID's time
        group_time = np.subtract(group.values.astype(np.float64).reshape((-1, 1)),
                                 np.array(values[:-1]).reshape((-1, 1)))
        group_time = group_time.reshape((-1, 1)).round(5)
        # print(group_time)

        #### id ###################################################
        group_id = list(map(f, name))
        if 16 in group_id:
            write_url('{},name to int error ID:{} to {},error result:{}'.format(filename, nam, name, group_id),
                      filename)
            return
        id_arra = np.zeros((len(indexs), 3), dtype=np.float64)
        id_arra[:, 0] = id_arra[:, 0] + group_id[0]
        id_arra[:, 1] = group_id[1] + id_arra[:, 1]
        id_arra[:, 2] = group_id[2] + id_arra[:, 2]

        # dlc ####################################################
        group_dlcs = data.loc[indexs, 2].values.astype(np.float64)
        group_dlcs = group_dlcs.reshape((-1, 1))

        # ###########  contents ##################################
        group_contents = data.loc[indexs, 3:].values
        contents = group_contents.copy()

        flag = 0
        np.set_printoptions(precision=3)

        # collect label and contents
        count = 0
        """处理message contents归一化"""
        for rows in group_contents:
            count += 1
            if count % 100000 == 0:
                writelog("{} pid processing current: {},name: {},index:{}".
                         format(pid, filename, name, count), filename)
            rows = rows.tolist()
            stop_cont = 8
            # 标签所在位置
            content = []
            if len(rows) == 8:

                for i, dat in enumerate(rows):
                    if dat == 'None':
                        stop_cont = i
                        break
                    else:
                        content.extend(list(map(f, dat)))
            else:
                writelog('error arise,id:{}\'s raw content lengh is {},details:{}'.format(name, len(rows), rows),
                         filename)
                return
            # completion of  name to content,from str to int
            if 8 - stop_cont:
                for i in range(8 - stop_cont):
                    content.extend([0, 0])
            # judge whether the content in ilgal such as has elem such as 16 ,or lengh is low than 16
            if len(content) == 16:
                if 16 in content:
                    writelog('error arise,id:{}\'s 16 in content,details:{}'.format(name, content), filename)
                    return
                pass
            else:
                writelog('error arise,id:{}\'s content lengh is {},details:{}'.format(name, len(content), content),
                         filename)
                return

            if flag == 0:
                contents = np.array(content).reshape((1, -1))
                flag = 1
            else:
                contents = np.concatenate((contents, np.array(content).reshape((1, -1))), axis=0)
        np.set_printoptions(suppress=True, precision=6)

        contents = np.concatenate((group_time, id_arra, group_dlcs, contents), axis=1).astype(np.float64)
        try:
            data1.loc[indexs, :] = contents.round(6)  # np.around(contents, decimals=5)
        except:
            print('error' * 8, '\n', contents[:5].round(6))
            print(data1.loc[indexs, :])
            print(contents.shape)
            print(len(indexs))
            return
    np.set_printoptions(suppress=True, precision=3)

    # print(data1.loc[:11, :])
    ############# write file ############################################
    # data1.to_csv(write_url, sep=' ',index=False, header=False, mode='a', float_format='%.2f')  # write_url
    data1.to_pickle(write_url, compression='zip')
    # writelog('finished:{} file named:{},total numbers:{},{}'.format(pid,filename,num,'*'*40),filename)
    try:
        writelog('finished:{} file named:{},total numbers:{},new shape:{},{}'.
                 format(pid, filename, num, data1.shape, '*' * 40), filename)
        print('finished:{} file named:{},total numbers:{},new shape:{},{}'.
              format(pid, filename, num, data1.shape, '*' * 40))
    except NameError:
        writelog('Error at finishe writelog func,is finished', filename)
    writelog('{} finished at:{}'.format(filename, time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))),
             filename)

def inter_id_diff64(url):
    """
    func:输入,hacking dataset原始数据集,ID为四位的内容.
    计算相邻的相同ID时间间隔,将16进制转化为10进制,补全内容,没有归一化.
    :param url:
    :return:
    """
    print(inter_id_diff64.__name__)
    pid = os.getpid()
    read_url = url[0]
    write_url = url[1]
    filename = os.path.basename(read_url)
    filename = os.path.splitext(filename)[0]
    # print(filename)
    # data = pd.read_pickle(read_url, compression='zip')
    data = pd.read_csv(read_url,sep=None,delimiter=' ',dtype=np.str,header=None,engine='python',encoding='utf-8')
    print('ready!!!shape:{}'.format(data.shape))
    # print(data)
    # print(data.iloc[4,9].__class__,data.iloc[4,8].__class__)
    # print(data.iloc[4,9]=='nan')
    #
    # exit()
    columns = [i for i in range(data.shape[1])]
    data.columns = columns
    if data.shape[1] > 11:
        print(filename, data.shape)
        print(data.loc[:10, :])
        return
    # data1 = data.copy()
    writelog('{} pid processing {}'.format(pid, filename))
    """扩充data规模"""
    # df2 = pd.DataFrame([[i for i in range(10)]], index=[0], columns=range(11, 21))
    # data1 = pd.concat([data1, df2], axis=1)
    # # 设置目标数据承载空间
    # writelog('{} reshape before:{},after:{}'.format(filename, data.shape, data1.shape), filename)
    # print('{} reshape before:{},after:{}'.format(filename, data.shape, data1.shape))
    num = data.shape[0]
    data.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列

    l = data.shape[0]
    row_ceil = math.ceil(l/64)
    row_floor = math.floor(l/64)
    print('row_ceil:',row_ceil)

    for start in range(row_ceil):
        if start == row_floor:
            block = data.iloc[start:, :]
        else:
            block = data.iloc[start:start+64,:]

        names = block.loc[:,1].values.tolist()
        contentss = block.loc[:,3:]

        badname = []
        names_ = []
        fl_ag = 0

        group_id = np.empty((1,3))
        """ 处理id """
        for i,name in enumerate(names):
            if name == 'None' or name == None:
                writelog('{} None Error,id is:{}'.format(filename, name), filename)
                fl_ag = 1

            elif len(name) == 4:

                if name[0] == '0':
                    try:
                        group_id = list(map(f, name[1:4]))
                        if 16 in group_id:
                            write_url('{},name to int error ID:{} to {},error result:{}'.format(filename, name, group_id), filename)
                        names_.append(group_id)
                        # writelog('ID:{} truncated as {}'.format(nam, name), filename)
                    except IndexError:
                        writelog('{} IndexError: {}'.format(filename, name), filename)
                        return
                else:
                    badname.append(name)
                    fl_ag = 1
            else:
                fl_ag = 1
                badname.append(name)
                # name = name+'0'
                writelog('{} special error ID:{}'.format(filename, name), filename)

            if fl_ag:
                writelog('error ID has {} at No.{} block,index is {}'.format(len(badname),start,start+i), filename)
                badname = np.array(badname).reshape((-1,8))
                u_bad = '/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/2/logs/badid.txt'
                np.savetxt(u_bad,badname,fmt='%d',delimiter=' ',encoding='utf-8')
                return
        group_id = np.array(names_).reshape((-1,3))

        ##### Time ##############################################
        values = block.loc[:,0].values.tolist()
        values.insert(0, values[0])
        np.set_printoptions(suppress=True, precision=5)

        # subtraction of adjacent same ID's time
        group_time = np.subtract(block.loc[:,0].values.astype(np.float64).reshape((-1, 1)),
                                 np.array(values[:-1]).reshape((-1, 1)).astype(np.float64))
        group_time = group_time.reshape((-1, 1)).round(5)
        # print(group_time)

        # dlc ####################################################
        group_dlcs = block.loc[:, 2].values.astype(np.float64)
        group_dlcs = group_dlcs.reshape((-1, 1))

        # ###########  contents ##################################
        group_contents = contentss.values
        contents = group_contents.copy()

        flag = 0
        np.set_printoptions(precision=3)

        # collect label and contents
        count = 0
        """处理message contents归一化"""
        for j,rows in enumerate(group_contents):
            count += 1
            rows = rows.tolist()
            stop_cont = 8
            # 标签所在位置
            content = []
            if len(rows) == 8:

                for i, dat in enumerate(rows):
                    if dat == 'None' or dat == None:
                        stop_cont = i
                        break
                    else:
                        content.extend(list(map(f, dat)))
            else:
                writelog('error arise at :{},raw content lengh is {},details:{}'.format(start+j, rows),
                         filename)
                return
            # completion of  name to content,from str to int
            if 8 - stop_cont:
                for i in range(8 - stop_cont):
                    content.extend([0., 0.])
            # judge whether the content in ilgal such as has elem such as 16 ,or lengh is low than 16
            if len(content) == 16:
                if 16 in content:
                    writelog('error arise at :{}, 16 in content,details:{}'.format(start+j, content), filename)
                    return
                pass
            else:
                writelog('error arise at :{}, content lengh is {},details:{}'.format(start+j, len(content), content),
                         filename)
                return

            if flag == 0:
                contents = np.array(content).reshape((1, -1))
                flag = 1
            else:
                contents = np.concatenate((contents, np.array(content).reshape((1, -1))), axis=0)
        np.set_printoptions(suppress=True, precision=6)
        # print(group_time.reshape, group_id.shape, group_dlcs.shape, contents.shape)
        # exit()
        contents = np.concatenate((group_time, group_id, group_dlcs, contents), axis=1).astype(np.float64)

        datt = pd.DataFrame(contents)
        if datt.shape[1] != 21:
            writelog('error start at {},block:\n'.format(i+start*64),filename)
            writelog('error start at {},block:\n'.format(i+start*64))
            print(datt.loc[:15,:])
            print(datt.loc[15:30,:])
            print(datt.loc[30:45,:])
            print(datt.loc[45:,:])
            return
        if start%6000==0:
            print('iter at {},shape:{}'.format(start,datt.shape))
        datt.to_csv(write_url, sep=' ',index=False, header=False, mode='a', float_format='%.4f')
        # exit()

        # if start == row_floor:
        #     data1.iloc[start:, :] = contents
        # else:
        #     data1.iloc[start:start+64,:] = contents
    # np.set_printoptions(suppress=True, precision=3)

    # print(data1.loc[:11, :])
    ############# write file ############################################
    # data1.to_csv(write_url, sep=' ',index=False, header=False, mode='a', float_format='%.2f')  # write_url
    # data1.to_pickle(write_url, compression='zip')
    # writelog('finished:{} file named:{},total numbers:{},{}'.format(pid,filename,num,'*'*40),filename)
    try:
        writelog('finished:{} file named:{},total numbers:{},{}'.
                 format(pid, filename, num, '*' * 40), filename)
        writelog('finished:{} file named:{},total numbers:{},{}'.
                 format(pid, filename, num, '*' * 40))
    except NameError:
        writelog('Error at finishe writelog func,is finished', filename)
    writelog('{} finished at:{}'.format(filename, time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))),
             filename)


def Mutual_time_interval(data,mark='same'):
    """
    func: take time interval
    :param data: 1D array-like date,[[1,2,3,4]],if mark=='ignore',data is 2D array-like data or DataFrame
    :param mark: 'same':same ID timeinterval,'ignore': ignore the regular of same ID
    :return: not
    """
    flag = 0

    if mark == 'ignore':
        print('run at{},part of {}'.format(Mutual_time_interval.__name__,mark))
        if data.__class__ == np.ndarray:
            data = data.astype(np.float64).reshape((-1,1))
        elif data.__class__ == pd.core.series.Series:
            data = data.values.astype(np.float64).reshape((-1,1))
        print('source data shape:{}'.format(data.shape))
        data_ = data.copy().tolist()
        data_.insert(0,data_[0])
        result = np.subtract(data,np.array(data_[:-1]).reshape((-1,1)))
        print(result.shape)
        print(' finished')
        print('max|min:{}|{}'.format(np.max(result,1),np.min(result,1)))
        return result

    else:
        print('run at{},part of {}'.format(Mutual_time_interval.__name__,mark),end=',')
        badids = []
        ids = []
        if data.__class__ == pd.core.frame.DataFrame:
            data.astype(np.str)
            print('source data shape:{}'.format(data.shape))
            # data.dtypes = np.str
        elif data.__class__ == np.ndarray:
            # print(data[:10,:])
            data = pd.DataFrame(data.astype(np.str).reshape((-1,2)),dtype=np.str)
            print('data.shape:{}'.format(data.shape ))

        else:
            print('error')
            return

        groupes = data[0].groupby(data[1])
        names = list(groupes.groups.keys())

        data_ = data.copy()
        c_max = 0.
        c_min = 1000000.
        print('len of names:%d'%len(names))
        count = 0
        for name, group in groupes:
            count +=1
            if count%200==0:
                print('current at:{}'.format(count))
            indexs = group.index.tolist()
            if len(name)>5 or len(name)<=3:
                print('error:{}'.format(name))
                flag = 1
                badids.append(name)
                continue
            elif name[0] != '0':
                badids.append(name)
                flag = 1
                continue
            else:
                if name[1:] in ids:
                    print('error:{}'.format(name))
                    flag = 1
                    print('len of error indexs:{}'.format(len(indexs)))
                    continue
                else:
                    ids.append(name[1:])
            # print(len(indexs))
            values = group.values.astype(np.float64).tolist()
            values.insert(0, values[0])
            np.set_printoptions(suppress=True, precision=5)
            group_time = np.subtract(group.values.astype(np.float64),
                                     np.array(values[:-1]))
            # group_time = group_time.round(6)
            data_.loc[indexs, 0] = group_time
            # group_time = group_time.reshape((-1, 1))
            if max(group_time.tolist()) > c_max:
                c_max = max(group_time.tolist())
            if min(group_time.tolist()) < c_min:
                c_min = min(group_time.tolist())
        if flag:
            print('the numbers of name before and after:{}|{}'.format(len(names),len(ids)))
            print('the numbers of error id ,len={}:{}'.format(len(badids),badids))
            print('error! no finished')
            return
        arr_time = data_.loc[:,0].values
        # print('result shape:{}, max|min:{}|{},max||min:{}||{}'
        #       .format(arr_time.shape, np.max(arr_time, 0), np.min(arr_time, 0),c_max,c_min))
        print('result shape:{},max||min:{}||{}'.format(arr_time.shape,c_max,c_min))

        print(' finished')

        return data_.loc[:,0]


def scalarToNegtiveAndPositive(url):
    raw = url[0]
    des = url[1]
    print(des)
    filename = os.path.splitext(os.path.basename(raw))[0]
    data = pd.read_pickle(raw, compression='zip')
    datashape = data.shape
    print(filename, datashape)
    # print(data.loc[:10,:])
    # rows = datashape[0]
    size = 64
    rows = 128
    columns = datashape[1]
    row_ceil = math.ceil(rows / size)
    row_floor = math.floor(rows / size)
    print('data ceil rows:', row_ceil,row_floor)
    c_max = [0 for _ in range(columns)]
    count = 0
    for start in range(row_ceil):
        print(start)
        scaler = MinMaxScaler(feature_range=(-1, 1))  # copy=True,
        if start == row_floor:
            scale_a = scaler.fit_transform(data.loc[start*size:, :].values.astype(np.float64)).reshape((-1, columns))
            print(data.loc[start*size:, :])

            # block.set_index(keys=, drop=True)
        else:
            scale_a = scaler.fit_transform(data.loc[start*size:start*size + size - 1, :].values.astype(np.float64)).reshape((-1, columns))
            print(data.loc[start*size:start*size + size - 1, :])
        c_max = list(map(f, c_max, scaler.data_max_))
        print('scale_a.shape:%s' % scale_a.shape)
        print(scale_a)
        count += scale_a.shape[0]
        if start % 1000 == 0:
            print('scale_a.shape:%s' % scale_a.shape, end=',')
            print('current at:%d' % start, end=',')
            print('count:%d' % count,end=',')
            print('filename:%s'%filename)

        # scale_a = np.around(scale_a, decimals=3)
        scale_a = pd.DataFrame(scale_a)
        scale_a.to_csv(des, sep=' ', index=False, header=False, mode='a', encoding='utf-8', float_format='%.4f',
                       index_label=None)
    # data.to_pickle(des,compression='zip')
    print('*' * 40)

# f = lambda x,y:x if operator.ge(x,y) else y


if __name__ == "__main__":
    print('program  started at:', time.asctime( time.localtime(time.time()) ))#time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))

    pd.set_eng_float_format(7,True)
    pd.set_option('precision', 7)


    # l = '/home/gjj/dataset/hacking/ignore_ID_diff_time'
    # ls = [os.path.join(l,f) for f in os.listdir(l)]
    # data = pd.read_pickle(ls[0],compression='zip')
    # print(data.loc[:10,:])
    # pd.set_option('chop_threshold', .5)

    # source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/2/data/Attack_free_dataset2.pkl"
    # dire_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/"
    # dire_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/from_raw_change_scaler/2/data"

    # print('program  start at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    # # print('data from :%s'%source_addr)
    # print()
    # os.chdir(os.path.dirname(dire_addr))
    # dire_url = os.path.join(dire_addr, 'Attack_free_dataset_64.txt')
    # print("\ncurrent at:{}".format(os.getcwd()))
    # print()
    """pkl to txt"""
    # dires = '/home/gjj/dataset/time_interval'
    # for f in os.listdir(dires):
    #     source_addr = os.path.join(dires,f)
    #     print('data from :%s' % source_addr)
    #     # print('data write at:%s' % dire_url)
    #     # exit()
    #     # raw_to_hackingdata((source_urls[0],dire_urls[0]))
    #     data = pd.read_pickle(source_addr, compression='zip')
    #     data = data.astype(np.float64)
    #     # data = pd.read_csv(source_addr,sep=None,delimiter=' ',dtype=np.str,header=None,engine='python',encoding='utf-8')
    #     print(data.shape)
    #     print(data.loc[:10, :])
    #     dires = '/home/gjj/dataset/time_interval'
    #     ignore = '_ignore_id_time_interval.pkl'
    #     same = '_same_id_time_interval.pkl'
    #     url = os.path.join(dires, os.path.splitext(f)[0] + '.txt')
    #     # print(url)
    #     # exit()
    #     # print(data[0].values[:10])
    #     # res = Mutual_time_interval(data.loc[:,0:1].values,'ll')
    #     # res = Mutual_time_interval(data.loc[:, 0].values)
    #     # res = pd.DataFrame(res)
    #     # data.to_csv(os.path.splitext(source_addr)[0]+'.txt', sep=' ',index=False, header=False, mode='w',encoding='utf-8')# float_format='%.4f'
    #     data.to_csv(url, sep=' ',index=False, header=False, mode='w',encoding='utf-8',float_format='%.6f')
    # print('program  finished at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))

    """cal time"""
    # source_addr = "/home/gjj/dataset/pure_data/instrusion/Attack_free_dataset2.pkl"

    # source_addr = "/home/gjj/PycharmProjects/ADA/raw_data/car-hacking-intrusion-dataset/origin-data"
    # dest = '/home/gjj/dataset/hacking/Intervaltime'
    # # print('data from :%s' % source_addr)
    # ignore = '_ignore_id_time_interval'
    # same = '_same_id_time_interval'
    # raws = [os.path.join(source_addr,f) for f in os.listdir(source_addr) if '.pkl' in f]
    # # print(raws)
    # # exit()
    # # destes = [os.path.join(dest, os.path.splitext(os.path.basename(f))[0]+ ignore) for f in os.listdir(source_addr) if '.pkl' in f]
    # # raw_to_hackingdata((source_urls[0],dire_urls[0]))
    # for source in raws:
    #     des = os.path.join(dest, 'INTERVAL_time',os.path.splitext(os.path.basename(source))[0] + same+'.pkl')
    #     print('source: {}'.format(source))
    #     print('des:{}'.format(des))
    #     data = pd.read_pickle(source,compression='zip')
    #     # data = pd.read_csv(source_addr,sep=None,delimiter=' ',dtype=np.str,header=None,engine='python',encoding='utf-8')
    #     filename = os.path.splitext(os.path.basename(source))[0]
    #     res = Mutual_time_interval(data.loc[:,:1],'same')
    #     if res.__class__ == pd.core.series.Series or res.__class__ == np.ndarray:
    #         pass
    #     else:
    #         continue
    #     res = pd.DataFrame(res)
    #     # data.to_csv(os.path.splitext(source_addr)[0]+'.txt', sep=' ',index=False, header=False, mode='w',encoding='utf-8')# float_format='%.4f'
    #
    #     print('{} shape:{},result shape:{}'.format(filename,data.shape,res.shape))
    #     # print(res)
    #     # exit()
    #     # exit()
    #     res.to_pickle(des, compression='zip')
    #     url = os.path.splitext(des)[0] +'.txt'
    #     res.to_csv(url,sep=' ',index=None,columns=None,header=None,encoding='utf-8',mode='w',float_format='%.7f')
    #     print('*'*60,end='\n\n')

    # url = os.path.join(dires,os.path.splitext(os.path.basename(source_addr))[0]+ignore)
    # for
    # print(url)
    # exit()
    # print(data[0].values[:10])
    # res = Mutual_time_interval(data.loc[:,0:1].values,'ll')
    # res = Mutual_time_interval(data.loc[:,0].values)
    # res = pd.DataFrame(res)
    # # data.to_csv(os.path.splitext(source_addr)[0]+'.txt', sep=' ',index=False, header=False, mode='w',encoding='utf-8')# float_format='%.4f'
    #
    # res.to_pickle(url,compression='zip')
    #
    # data.to_pickle(os.path.splitext(source_addr)[0]+'.txt',compression='zip')
    # inter_id_diff64((source_addr, dire_url))

    """construct to ignore dataset"""

    # header = '/home/gjj/dataset/hacking/Intervaltime/ignoreID'
    # trail = '/home/gjj/dataset/hacking/same_ID_diff_time'
    # dest = '/home/gjj/dataset/hacking/ignore_ID_diff_time'
    # source = '/home/gjj/dataset/same_ID_diff_time'
    # dest = '/home/gjj/dataset/sameScalarNegtiveToPasitive'
    # heades = [os.path.join(header,f) for f in os.listdir(header) if '.pkl' in f]
    # trailes = [os.path.join(trail,f) for f in os.listdir(trail) if '.pkl' in f]
    # source = '/home/gjj/dataset/same_ID_diff_time'

    # source = '/home/gjj/dataset/hacking/same_ID_diff_time'
    # # dest = '/home/gjj/dataset/hacking/ignore_ID_diff_time'
    # t = '/home/gjj/dataset/hacking/Intervaltime/sameID'
    # dest = '/home/gjj/dataset/hacking/SAME_ID_diff_time'
    #
    # ignore = '_ignore_id_time_interval.pkl'
    # same = '_same_id_time_interval.pkl'
    #
    # bodies = [os.path.join(source,f) for f in os.listdir(source) if '.pkl' in f]
    # destes = [os.path.join(dest,f) for f in os.listdir(source) if '.pkl' in f]
    # heads = [os.path.join(t,os.path.splitext(f)[0] + same) for f in os.listdir(source) if '.pkl' in f]
    #
    # for body_, head_,des_ in list(zip(bodies,heads,destes)):
    #     url = os.path.splitext(des_)[0]+'.txt'
    #     if os.path.exists(url):
    #         continue
    #     body = pd.read_pickle(body_,compression='zip')
    #     head = pd.read_pickle(head_,compression='zip')
    #
    #     data = pd.concat([head, body.loc[:,1:]],axis=1).astype(np.float64)
    #
    #     print('head url:{},shape:{}'.format(head_,head.shape))
    #     print('body url:{},shape:{}'.format(body_,body.shape))
    #
    #     data.to_pickle(des_,compression='zip')
    #     print('dest pkl:{},shape:{}'.format(des_,data.shape))#
    #     print('dest txt:{}'.format(url))
    #
    #     print(data.loc[:10,:],end='\n\n')
    #
    #     data.to_csv(url,sep=' ',index=False, header=False, mode='w',encoding='utf-8',float_format='%.6f')#,float_format='%.8f'


    """scalar to (-1,1) or (0,1) batch size scalar"""
    # source = '/home/gjj/dataset/same_ID_diff_time'
    # t = '/home/gjj/dataset/time_interval'
    # source = '/home/gjj/dataset/hacking/ignore_ID_diff_time'
    # # dest = '/home/gjj/dataset/scalarNegtiveToPasitive'
    # source = '/home/gjj/dataset/same_ID_diff_time'
    # source = '/home/gjj/dataset/hacking/SAME_ID_diff_time'
    # dest = '/home/gjj/dataset/hacking/ignoreBatchScalarNegtiveToPositive'
    # dest = '/home/gjj/dataset/hacking/sameBatchScalarNegtiveToPositive'
    # raws = [os.path.join(source,f) for f in os.listdir(source) if '.pkl' in f]
    # # destes = [os.path.join(dest,f) for f in os.listdir(source) if '.pkl' in f]
    # # print(raws)
    # # print(destes)
    # # exit()
    # f = lambda x,y:x if operator.ge(x,y) else y
    # for raw in raws:#list(zip(raws,destes)):
    #     des = os.path.join(dest,os.path.basename(raw))
    #     print('source:%s'%raw)
    #     print('des:%s'%des)
    #     # continue
    #     data = pd.read_pickle(raw, compression='zip')
    #     # _data = data.copy()
    #     datashape = data.shape
    #     filename = os.path.splitext(os.path.basename(raw))[0]
    #     print(filename, datashape,end=',')
    #     # print(data.loc[:10,:])
    #     # exit()
    #     rows = datashape[0]
    #     size = 64
    #     # rows = 128
    #     columns = datashape[1]
    #     row_ceil = math.ceil(rows / size)
    #     row_floor = math.floor(rows / size)
    #     print('data rows ceil | floor:{}|{}'.format( row_ceil, row_floor),end='\n\n')
    #     c_max = [0 for _ in range(columns-1)]
    #     # print('len of c_max:',len(c_max))
    #     count = 0
    #
    #     url = os.path.splitext(des)[0] + '.txt'
    #     for start in range(row_ceil):
    #         # print(start)
    #         scaler = MinMaxScaler(feature_range=(-1, 1))  # copy=True,
    #
    #         if start == row_floor:
    #             data_ = pd.DataFrame(data.loc[start * size:, :].values)
    #         else:
    #             data_ = pd.DataFrame(data.loc[start * size:start * size + size - 1, :].values)
    #         scale_a = scaler.fit_transform(data_.loc[:, :columns-2].values.astype(np.float64)).reshape((-1, columns-1))
    #         data_.loc[:, :columns - 2] = scale_a
    #         c_max = list(map(f, c_max, scaler.data_max_))
    #
    #         if start % 1000 == 0:
    #             print('data_:{}'.format(data_.shape),end=',')
    #             print('scale_a.shape:{}'.format(scale_a.shape), end=',')
    #             print('current at:%d' % start, end=',')
    #             print('count:%d' % count, end=',')
    #             print('filename:%s' % filename)
    #
    #         data_.to_csv(url, sep=' ', index=False, header=False, mode='a', encoding='utf-8', float_format='%.6f',
    #                        index_label=None)
    #     #     if start == row_floor:
    #     #         _data.loc[start * size:, :]= data_.values
    #     #     else:
    #     #         _data.loc[start * size:start * size + size - 1, :] = data_.values
    #         count += scale_a.shape[0]
    #     #
    #     # _data.to_pickle(des,compression='zip')
    #     # print(_data.loc[data.shape[0]-10:,:])
    #     print('c_max:{}'.format(c_max))
    #     print('*'*60,end='\n\n')

    #     # exit()


    # pool = mp.Pool(processes=len(raws))
    # pool.map_async(scalarToNegtiveAndPositive,zip(raws,destes,))
    # pool.close()
    # pool.join()

    """scalar to -1,1, full saclar """
    source = '/home/gjj/dataset/hacking/SAME_ID_diff_time'
    # dest = '/home/gjj/dataset/hacking/sameFullScalarZeroToPositive'
    dest = '/home/gjj/dataset/hacking/SAMEFullScalarZeroToPositive'
    raws = [os.path.join(source,f) for f in os.listdir(source) if '.pkl' in f]
    for raw in raws:#list(zip(raws,destes)):
        des = os.path.join(dest,os.path.basename(raw))
        url = os.path.splitext(des)[0]+'.txt'
    
        if os.path.exists(des) and os.path.exists(url):
            continue

        print('source:%s'%raw)
        print('des:%s'%des)
        # continue
        data = pd.read_pickle(raw, compression='zip')
        # _data = data.copy()
        datashape = data.shape
        filename = os.path.splitext(os.path.basename(raw))[0]
        print(filename, datashape,end=',')
        # print(data.loc[:10,:])
        columns = data.shape[1]
        c_max = [0 for _ in range(columns-1)]

        scaler = MinMaxScaler(feature_range=(0, 1))  # copy=True,

        scale_a = scaler.fit_transform(data.loc[:, :columns-2].values.astype(np.float64)).reshape((-1, columns-1))
        data.loc[:, :columns - 2] = scale_a
        # if scaler.data_max_ >c_max:
        #     c_max = scaler.data_max_
        c_max = list(map(d, c_max, scaler.data_max_))

        print('data:{}'.format(data.shape),end=',')
        print('scale_a.shape:{}'.format(scale_a.shape), end=',')
        if os.path.exists(des):
            pass
        else:
            data.to_pickle(des,compression='zip')

        if os.path.exists(url):
            pass
        else:
            data.to_csv(url, sep=' ', index=False, header=False, mode='w', encoding='utf-8', float_format='%.6f',
                           index_label=None)
        # print(_data.loc[data.shape[0]-10:,:])
        print()
        print('c_max:{}'.format(c_max))
        print(data.loc[:10, :])
        print('*'*60,end='\n\n')

    """contact time and the other"""
    # header = '/home/gjj/dataset/hacking/Intervaltime/ignoreID'
    # trail = '/home/gjj/dataset/hacking/same_ID_diff_time'
    # dest = '/home/gjj/dataset/hacking/ignore_ID_diff_time'
    # # source = '/home/gjj/dataset/ignore_ID_diff_time'
    # # dest = '/home/gjj/dataset/scalarNegtiveToPasitive'
    # # source = '/home/gjj/dataset/same_ID_diff_time'
    # # dest = '/home/gjj/dataset/sameScalarNegtiveToPasitive'
    # heades = [os.path.join(header,f) for f in os.listdir(header) if '.pkl' in f]
    # trailes = [os.path.join(trail,f) for f in os.listdir(trail) if '.pkl' in f]
    #
    # print(heades)
    # print(trailes)
    # exit()
    # exit()
    # for content,head in list(zip(raws,destes)):
    #     data = pd.read_csv(content,sep=None,delimiter=' ',dtype=np.float64,header=None,engine='python',encoding='utf-8')
    #     data.to_pickle(head,compression='zip')
    #     print(os.path.splitext(os.path.basename(content))[0],end=','); print(data.shape)
    """txt to pkl"""
    # source = '/home/gjj/dataset/ignore_ID_diff_time'
    # dest = '/home/gjj/dataset/scalarNegtiveToPasitive'
    # dest = '/home/gjj/dataset/sameScalarNegtiveToPasitive'
    # dest = '/home/gjj/dataset/hacking/same_ID_diff_time'
    # raws = [os.path.join(dest,f) for f in os.listdir(dest) if '.txt' in f]
    # destes = [os.path.splitext(f)[0]+'.pkl' for f in raws if '.txt' in f]

    # dest = '/home/gjj/dataset/hacking/ignoreBatchScalarNegtiveToPositive'

    # dest = '/home/gjj/dataset/hacking/sameFullScalarZeroToPositive'

    # dest = '/home/gjj/dataset/hacking/sameBatchScalarNegtiveToPositive'
    # raws = [os.path.join(dest,f) for f in os.listdir(dest) if '.txt' in f]
    #
    # # print(raws)
    # # exit()
    # # print(destes)
    # # exit()
    # for raw in raws:
    #     des = os.path.splitext(raw)[0]+'.pkl'
    #     if os.path.exists(des):
    #         continue
    #     print(des)
    #     try:
    #         data = pd.read_csv(raw,sep=None,delimiter=' ',dtype=np.float64,header=None,engine='python',encoding='utf-8')#,nrows=100)
    #     except OSError:
    #         print('{} arise OSError [Errno 5] Input/output error'.format(os.path.splitext((os.path.basename(des)))[0]))
    #         continue
    #     # print(data.loc[data.shape[0]-10:,:])
    #     # continue
    #     data.to_pickle(des, compression='zip')
    #     print(data.shape)
    #     print(data.loc[data.shape[0]-5:,:])
    #     print()

    # print(raws)
    # print(destes)
    print('program  finished at:', time.asctime( time.localtime(time.time()) ))#time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time()))

