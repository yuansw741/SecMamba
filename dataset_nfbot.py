# import dgl.nn as dglnn
# from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
# import dgl.function as fn
# from dgl.data.utils import load_graphs
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

# 创建标签编码器
label_encoder = LabelEncoder()

data = pd.read_csv(r'./dataset/raw/NF-BoT-IoT-v2_0_0.csv')
# ysw
# data = data.sample(frac=0.05,random_state = 123)


# data = data.sort_values('stime').reset_index(drop=True)
# data['stime'] = range(1, len(data) + 1)
#
# ts = data['stime']
# data.drop(columns=['subcategory','pkSeqID','stime','flgs','category','state','proto','seq'],inplace=True)

data.drop(columns=['Label'],inplace=True)
data.rename(columns={"Attack": "label"},inplace = True)
data['label'] = label_encoder.fit_transform(data['label'])
print("@@@@")
# print(data['time'])
# time = data['time']


data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)

data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']

# 尝试将 'sport' 列转换为数值类型（整型），无法转换的值会被设置为 NaN
data['L4_SRC_PORT'] = pd.to_numeric(data['L4_SRC_PORT'], errors='coerce')

# 检查哪些 'sport' 值无法转换（会显示为 NaN）
invalid_sport_values = data[data['L4_SRC_PORT'].isna()]

# 输出无法转换的行
print("无效的 sport 值行:")
print(invalid_sport_values['L4_SRC_PORT'])
data['L4_SRC_PORT'].fillna(0, inplace=True)
data['L4_SRC_PORT'] = data['L4_SRC_PORT'].astype(int)
sport_column=data['L4_SRC_PORT']





# data.drop(columns=['sport','dport'],inplace=True)
data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)

label = data.label
# data.drop(columns=['label','time'],inplace = True)
data.drop(columns=['label'],inplace = True)
scaler = StandardScaler()
encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL'])
encoder.fit(data, label)
data = encoder.transform(data)
data =  pd.concat([data, label], axis=1)
cols_to_norm = list(set(list(data.iloc[:, 2:].columns )) - set(list(['label'])) )
data[cols_to_norm] = scaler.fit_transform(data[cols_to_norm])
data['h'] = data[ cols_to_norm ].values.tolist()

print(data)

unique_ips = pd.concat([data['IPV4_SRC_ADDR'], data['IPV4_DST_ADDR']]).unique()
ip_to_int = {ip: idx for idx, ip in enumerate(unique_ips)}

data['IPV4_SRC_ADDR_int'] = data['IPV4_SRC_ADDR'].map(ip_to_int)
data['IPV4_DST_ADDR_int'] = data['IPV4_DST_ADDR'].map(ip_to_int)

data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR_int']
data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR_int']

data.drop(columns=['IPV4_SRC_ADDR_int','IPV4_DST_ADDR_int'],inplace=True)

# new_data = pd.concat([data['IPV4_SRC_ADDR'],data['IPV4_DST_ADDR'],time,data['label'],data['h']], axis=1)
new_data = pd.concat([data['IPV4_SRC_ADDR'],data['IPV4_DST_ADDR'],data['label'],data['h']], axis=1)

# print(time)
# print(new_data)

h_expanded = data['h'].apply(pd.Series)
data_expanded = pd.concat([new_data.drop(columns=['h']), h_expanded], axis=1)
data_expanded.to_csv('Final_NFBot_3.csv',index=False)