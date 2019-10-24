import pandas as pd
import os
import numpy as np
import sys
# 找出已经去掉timestamp等导读标签数据里面的正常数据块及每块的大小。
maxinum = 1500
def countNormalInternal(l1)->dict:
    start = 0
    dic = dict();
    print(len(ll1))
    for i in range(0,len(l1)):
        # print(l1[i])
        if l1[i] == "T":
            # print("duiduidui ")
            temp = i-start
            dic[start]=temp
            start = i
    return dic

if __name__ == "__main__":
    read_url = "F:\Yangyuanda\ADA\dealed_data\hacking_data"
    listdir = os.listdir(read_url)
    l1 = "\\".join([read_url,listdir[3]])
    print(l1)
    data = pd.read_csv(l1, sep=None, header=None, engine='python')
    # print(data.shape)
    ll1 = data[4].to_numpy(str,True).tolist()
    # print(type(ll1),ll1[0:10])
    dicc = countNormalInternal(ll1)
    print(max(dicc.values()))
    for i in dicc.keys():
        if dicc.get(i)>= maxinum:
            print("{} {}".format(i,dicc[i]))