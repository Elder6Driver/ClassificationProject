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

data['销售人员服务等级'].replace('SSA',3)
data['销售人员服务等级'].replace('SSB',2)
data['销售人员服务等级'].replace(-1,1)
#将部分列设置为onehot类型，这里将其放入onehotmatrix，并在原序列中去除
onehotmatrix=pd.get_dummies(data['公司类型']).as_matrix()
data.drop(['公司类型'],axis=1,inplace=True)

onehotmatrix = np.column_stack((onehotmatrix,pd.get_dummies(data['销售人员服务等级']).as_matrix()))
data.drop(['销售人员服务等级'],axis=1,inplace=True)

# ss = data.corr()
# ss.to_csv(r"C:\Users\maoyu\Desktop\newans.csv")

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


# Parameters设定学习率，整体迭代次数，每一批迭代的数量，每隔多少次显示下训练集准确率
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters设定输入输出隐层节点数目
n_hidden_1 = 20 # 1st layer number of features
n_hidden_2 = 15 # 2nd layer number of features
n_input = tfdatatrainx.shape[1] # MNIST data input (img shape: 28*28)
n_classes = tfdatatrainy.shape[1] # MNIST total classes (0-9 digits)


start = 0
end = batch_size
epochnow = 0

#初始化权重方法
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


#模型，用relu激活，增加了随机失活
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


x = tf.placeholder("float", [None, tfdatatrainx.shape[1]])
y = tf.placeholder("float", [None, tfdatatrainy.shape[1]])

w_h = init_weights([tfdatatrainx.shape[1], n_hidden_1])
w_h2 = init_weights([n_hidden_1, n_hidden_2])
w_o = init_weights([n_hidden_2, tfdatatrainy.shape[1]])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(x, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

#损失函数，交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, y))
#Adam下降
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
pred = tf.argmax(py_x, 1)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle

    #这里自己写了一个批量取数据的方法
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(tfdatatrainx.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            start = epochnow
            epochnow += batch_size
            if epochnow > tfdatatrainx.shape[0]:
                perm = np.arange(tfdatatrainx.shape[0])
                np.random.shuffle(perm)
                tfdatatrainx = tfdatatrainx[perm]
                tfdatatrainy = tfdatatrainy[perm]
                start = 0
                epochnow = batch_size
                assert epochnow <= tfdatatrainx.shape[0]
            end = epochnow
            batch_xs = tfdatatrainx[start:end]
            batch_ys = tfdatatrainy[start:end]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys,
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # # 这里试试看能不能把训练集准确度输出下
            # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # # Calculate accuracy
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # print("Accuracy:", accuracy.eval({x: tfdatatrainx, y: tfdatatrainy,
            #                               p_keep_input: 1, p_keep_hidden: 1}))
            # print(np.mean(w_h.eval()) )
            # print(np.std(w_h.eval()))
            # print(np.mean(w_h2.eval()))
            # print(np.std(w_h2.eval()))

            #每隔多少代计算下训练集的准确度
            print("train acc:",  np.mean(np.argmax(tfdatatrainy, axis=1) ==
                          sess.run(pred, feed_dict={x: tfdatatrainx,
                                                    p_keep_input: 1.0,
                                                    p_keep_hidden: 1.0})),
                  "test acc:" ,  np.mean(np.argmax(tfdatatesty, axis=1) ==
                             sess.run(pred, feed_dict={x: tfdatatestx,
                                                             p_keep_input: 1.0,
                                                             p_keep_hidden: 1.0})))

    #最后输出下测试集的准确度
    print("Optimization Finished!")

    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1.0), tf.argmax(y, 1.0))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: tfdatatestx, y: tfdatatesty}))
    print( np.mean(np.argmax(tfdatatesty, axis=1) ==
                             sess.run(pred, feed_dict={x: tfdatatestx,
                                                             p_keep_input: 1.0,
                                                             p_keep_hidden: 1.0})))


# Launch the graph in a session
# with tf.Session() as sess:
#     # you need to initialize all variables
#     tf.initialize_all_variables().run()
#
#     for i in range(100):
#         for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
#             sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end],
#                                           p_keep_input: 0.8, p_keep_hidden: 0.5})
#         print(i, np.mean(np.argmax(teY, axis=1) ==
#                          sess.run(predict_op, feed_dict={X: teX,
#                                                          p_keep_input: 1.0,
#                                                          p_keep_hidden: 1.0})))