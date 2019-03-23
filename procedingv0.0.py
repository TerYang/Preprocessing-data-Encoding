import pandas as pd
import numpy as np
import os

np.set_printoptions(suppress=True)

source_attr ="/home/gjj/PycharmProjects/ADA/dealed_data/instrusion_data/"
dire_attr = "/home/gjj/PycharmProjects/ADA/ID-P/"
attrs = os.listdir(source_attr)
# print(attrs)
if not os.path.exists(dire_attr):  # 如果保存模型参数的文件夹不存在则创建
    os.makedirs(dire_attr)

def accumulation_id(piece,attr):
    piece = piece.dropna(axis=1, how='all')  # how = ['any','all'] 丢弃空列
    # """ print(o1.dtypes)
    # 0    float64
    # 1     object
    # 2      int64
    # 3     object
    # """
    ids = o1.iloc[:,1].values.astype(np.str)#提取id列

    # ids = ["00100","11001","00000","10001"]
    for i, elem in enumerate(ids):
        # print(type(elem[0]))
        if elem[0] != '0' or elem[-2:] != "00":
            yield list(attr,elem)


if __name__ == "__main__":

    for attr in attrs:
        # if attr in had_dealed:
        #     continue
        print("current:",attr)
        # continue

        read_url = source_attr + attr
        dire_url = dire_attr + attr[0: attr.index('.')] + r"_ID.txt"
        # print(read_url,'\n',dire_url)
        # continue
        # np.set_printoptions(suppress=True,precision=6)
        data = pd.read_csv(read_url, sep=None, header=None, engine='python', chunksize=10000)#,dtype=np.str

        count = 0
        s1_fir_element = 0
        for o1 in data:
            count += 1
            if count % 10 == 0:
                print("loop:", count)
            accumulation_id(o1,attr)
        # exit()

            # print(type(ids))
            # print(ids.dtype)

            # arr = np.insert(o1.loc[:, 0].values, 0, s1_fir_element, axis=0)
            # arr1 = arr[1:] - arr[:-1]  # calculate the time difference between before and after
            # s1_fir_element = arr[-1]
            # # print(len(arr1),arr1)
            # # exit()
            # # # f = lambda x: int(x)
            # # s1_fir_element = arr[-1]
            # # # print(s1_fir_element,o1.iloc[-1,0])
            # # # exit()
            # # s1 = pd.Series(arr[:-1],index=np.arange(indexOfDF))
            # # s2 = pd.Series(arr[1:],index=np.arange(indexOfDF))
            # # # print(s1,s2)
            # # # exit()
            # # o1.loc[:,0] = s2.sub(s1).values#s2-s1
            # o1.loc[:, 0] = arr1  # set the columns 0 as the array arr1
            # print("o1.loc[:,0]\t",o1.loc[:,0])
            # print("values\t",o1.loc[:,0].values)
            # exit()
            # if count == 1:
            #     o1.loc[0, 0] = 0
            # if count==372:
            #     print(s1, s2)
            #     print(s2.sub(s1))
            # print(o1.loc[:,0])
            #     break
            # o1.to_csv(dire_url, sep=' ', index=False, header=False, mode='a')  # write_url