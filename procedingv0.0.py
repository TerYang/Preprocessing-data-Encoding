import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(suppress=False,precision=6)

# source_addr ="/home/gjj/PycharmProjects/ADA/ID-P/"
source_addr ="/home/gjj/PycharmProjects/ADA/ID-TIME_data/ID_TIME_instrusion_data/"
dire_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/"

addrs = os.listdir(source_addr)

print(addrs)
if not os.path.exists(dire_addr):  # 如果保存模型参数的文件夹不存在则创建
    os.makedirs(dire_addr)


def batch(np_or_series):
    """(value - mean )/rsqrt(variance) 数值减去均值之差除标准差,归一化后有负值"""
    # mean_value = np.mean(np_or_series)
    # std_value = np.std(np_or_series)
    # np_or_series = (np_or_series - mean_value)/std_value
    # print(np_or_series)
    """归一化在一定范围内"""
    # np.set_printoptions(suppress=True,precision=2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    print(np_or_series.shape)
    np_or_series = scaler.fit_transform(np_or_series.reshape(-1, 1))
    return np.around(np_or_series,decimals=2)#返回进度为小数点后两位

if __name__ == "__main__":

    for addr in addrs:
        print(addr)

        read_url = source_addr + addr
        dire_url = dire_addr + addr
        data = pd.read_csv(read_url, sep=None, header=None, engine='python', chunksize=100)
        # data = pd.read_csv(read_url, sep=None, header=None, engine='python')

        time_max=0
        """shape (100, 4),Index([0, 1, 2, 3], dtype='int64')
        dtypes: 1     object
                2      int64
                3     object
                dtype: object
        """
        # batch(data.loc[:, 0].values)



        # data = pd.read_csv(read_url,sep='\s+',delimiter=',', header=None, engine='python', chunksize=50,dtype=np.str)# shape(50, 1)
        for j, o1 in enumerate(data):
            if j != 0:
                break
            np1 = np.array(o1.loc[:8,3].values).astype(np.str)
            print(np.replace(np1,None,fffxxxxx))



            # o1 = pd.Series(o1.loc[:8,3])
            # # print(o1.map({'None':0}))
            # print(o1.replace([None],9))
        #     """max vule at o1 used func:DataFrame.max([axis, skipna, level, …])"""
        #     batch(o1.loc[:,0].values)
        #     o1_max = o1.max()
        #     max_value = o1_max.at[o1_max.index[0]]
        #     print(max_value)
        #     print(type(o1_max),o1_max.shape,o1_max.index)
        #     # print(type(o1),o1.index,o1.columns,o1.shape,o1.dtypes)
        #     # print(type(o1.loc[:,0]),o1.index,o1.columns,o1.shape,o1.loc[:,0].dtypes)
        #     # <class 'pandas.core.series.Series'> RangeIndex(start=0, stop=100, step=1) Int64Index([0, 1, 2, 3], dtype='int64') (100, 4) float64
        #     # print(type(o1.loc[:,3]),o1.index,o1.columns,o1.shape,o1.loc[:,3].dtypes)
        #     #<class 'pandas.core.series.Series'> RangeIndex(start=0, stop=100, step=1) Int64Index([0, 1, 2, 3], dtype='int64') (100, 4) object
        exit()
    # data = pd.read_csv(read_url, sep=None, header=None, dtype=np.str, engine='python', chunksize=10000)  # ,dtype=np.str