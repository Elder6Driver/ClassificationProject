#-*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#首先读取数据
data = pd.read_excel(r"C:\Users\maoyu\Desktop\useforpredict.xls" ,encoding="gb2312" )

#去掉一些不重要的列
data.drop(['公司所属目录'],axis=1,inplace=True)
data.drop(['公司状态'],axis=1,inplace=True)
data.drop(['折扣'],axis=1,inplace=True)
data.drop(['加强次数'],axis=1,inplace=True)
data.drop(['注册金额'],axis=1,inplace=True)
data.drop(['平均登录次数'],axis=1,inplace=True)
data.drop(['销售联系次数'],axis=1,inplace=True)


#这个一开始的想法是去掉公司ID为重复的项目，但是后来发现公司数据是连续的，所以去掉没什么卵子用
#data.drop_duplicates(['公司ID'] , inplace=True)

#去掉公司ID这个项
data.drop(['公司ID'],axis=1,inplace=True)

#为数据填充0
data.fillna(0,inplace=True)

# s = data.quantile(0.75)
# #询盘量
# data = data[data[data.columns[1]] < s[1]]
# #客服联系次数
# data = data[data[data.columns[4]] < s[4]]
# #销售联系次数
# data = data[data[data.columns[5]] < s[5]]
#
# #实际服务金额
# data = data[data[data.columns[7]] < s[7]]
# #数据罗盘登录次数
# data = data[data[data.columns[11]] < s[11]]
# #注册时长
# data = data[data[data.columns[13]] < s[13]]
# #最近一次登录时间
# data = data[data[data.columns[14]] < s[14]]

data['销售人员服务等级'].replace('SSA',3)
data['销售人员服务等级'].replace('SSB',2)
data['销售人员服务等级'].replace(-1,1)
#将部分列设置为onehot类型，这里将其放入onehotmatrix，并在原序列中去除
onehotmatrix=pd.get_dummies(data['公司类型']).as_matrix()
data.drop(['公司类型'],axis=1,inplace=True)

onehotmatrix = np.column_stack((onehotmatrix,pd.get_dummies(data['销售人员服务等级']).as_matrix()))
data.drop(['销售人员服务等级'],axis=1,inplace=True)

ss = data.corr()
ss.to_csv(r"C:\Users\maoyu\Desktop\newans.csv")

#归一化处理，这里使用的是（当前值-最小值）/范围 这个公式
maxseri = data.max()
minseri = data.min()
subseri = maxseri - minseri
# data.xi[622644452,'加强次数']
data = (data - minseri)/subseri
# data = data.join(frameonehot)
#print(data)
#data.to_csv(r"C:\Users\maoyu\Desktop\WHAT.csv")
#print(subseri)
#print(data - minseri)
#print(data)
# data = data[32:51]

#接下来讲data分为训练集和测试集，并提取出自变量和因变量

totalrow = data["是否续约"].count()
#totalcol = data[:1].count(axis=1)

#通过行数产生了一个随机的list , trainindex表示训练集索引，testindex表示测试集索引
lists = np.random.permutation(totalrow)
trainnum = int( totalrow * 0.7)
trainindex = lists[:trainnum]
testindex = lists[trainnum:]

#再DataFrame上面对数据集进行重新排列
datatrainx = data.drop(['是否续约'],axis=1).ix[data.index[trainindex]]
datatrainy = data['是否续约'].ix[data.index[trainindex]]
# datatrainx.to_csv(r"C:\Users\maoyu\Desktop\aaa.csv")
# datatrainy.to_csv(r"C:\Users\maoyu\Desktop\bbb.csv")
datatestx = data.drop(['是否续约'],axis=1).ix[data.index[testindex]]
datatesty = data['是否续约'].ix[data.index[testindex]]

#onthot列的重新排列
onthottrainx = onehotmatrix[trainindex,:]
onthottestx = onehotmatrix[testindex,:]

#将得到的数据转化为matrix，然后加入onehot列
tfdatatrainx = np.column_stack((datatrainx.as_matrix(), onthottrainx))
tfdatatrainy = datatrainy.as_matrix()

#这里需要对得到的内容转至下(这里做了下onehot的转换，其实没必要这么麻烦)
tfdatatrainy = np.array([tfdatatrainy,np.ones(tfdatatrainy.shape[0]) - tfdatatrainy]).T

tfdatatestx = np.column_stack((datatestx.as_matrix(), onthottestx))
tfdatatesty = datatesty.as_matrix()

tfdatatesty = np.array([tfdatatesty,np.ones(tfdatatesty.shape[0]) - tfdatatesty]).T


#这里想办法去掉0.95分位数以上的异常值

# s = data.quantile(0.75)
# #询盘量
# data = data[data[data.columns[1]] < s[1]]
# #客服联系次数
# data = data[data[data.columns[4]] < s[4]]
# #销售联系次数
# data = data[data[data.columns[5]] < s[5]]
#
# #实际服务金额
# data = data[data[data.columns[7]] < s[7]]
# #数据罗盘登录次数
# data = data[data[data.columns[11]] < s[11]]
# #注册时长
# data = data[data[data.columns[13]] < s[13]]
# #最近一次登录时间
# data = data[data[data.columns[14]] < s[14]]