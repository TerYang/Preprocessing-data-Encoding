import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import MultiLabelBinarizer as mlb
"""
OneHotEncoder(n_values=’auto’, 
categorical_features=’all’,  dtype=<class ‘numpy.float64’>,
sparse=True,  handle_unknown=’error’)
"""
# lianjia_df = pd.DataFrame({'Elevator':[1,2],'Renovation':[4,5]},dtype=np.float32,copy=True)
# print(lianjia_df.values)
# print(lianjia_df['Elevator'])
# l = pd.get_dummies(lianjia_df['Elevator'])#独热编码方法
# print(l)


x = np.random.uniform(1,10,[3,5]).astype(np.int32)
y = np.arange(1,10,0.5)

# print(x)
# # print(y)

# encoder = ohe(sparse=False)#指定结果是否稀疏
# encoder.fit(x)
# print(encoder.active_features_)
# print(encoder.feature_indices_)
# print(encoder.n_values_)
# print(encoder.transform([[1,2,3,4,5]]))

encoder = ohe(sparse=False)#指定结果是否稀疏后者transform .toarray()
encoder.fit([[0,0,3],[1,1,0],[0,2,1],[1,0,2]])
arra = encoder.transform([[0,1,3]])
print(arra)