import os
import sys
import math
import numpy
import pandas as pd
import paddle
import paddle.fluid as fluid
from sklearn.utils import shuffle
from model import ActorModel


BATCH_SIZE = 20

raw_dataset = pd.DataFrame()
for file in os.listdir('train_data'):
    if 'train_data' in file:
        raw_dataset = pd.concat([raw_dataset, pd.read_csv('train_data/' + file)], ignore_index=True)

raw_dataset = shuffle(raw_dataset, random_state=1)
test_proportion = 0.05
test_idx = int(len(raw_dataset) * test_proportion)
train_dataset = raw_dataset.iloc[test_idx:, :].tolist()
test_dataset = raw_dataset.iloc[:test_idx, :].tolist()


def reader_creator(train_data):
    def reader():
        for d in train_data:
            yield d[:-2], d[-2:]
    return reader


train_reader = fluid.io.batch(
    fluid.io.shuffle(
        reader_creator(train_dataset), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = fluid.io.batch(
    fluid.io.shuffle(
        reader_creator(test_dataset), buf_size=500),
    batch_size=BATCH_SIZE)


def policy():
    input = fluid.data(name='state', shape=[None, 11], dtype='float32')
    hidden1 = fluid.layers.fc(input=input, size=300, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=600, act='relu')
    prediction = fluid.layers.fc(name='action', input=hidden2, size=2, act='sigmoid')
    return prediction


def train_program(model):
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.data(name='label', shape=[None, 2], dtype='int64')

    # predict = softmax_regression() # 取消注释将使用 Softmax回归
    # predict = multilayer_perceptron() # 取消注释将使用 多层感知器
    predict = policy()  # 取消注释将使用 LeNet5卷积神经网络

    # 使用类均方差函数计算predict和label之间的损失函数
    cost = fluid.layers.square_error_cost(input=predict, label=label)
    # 计算平均损失
    avg_cost = fluid.layers.mean(cost)

    return predict, avg_cost


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


def event_handler(pass_id, batch_id, cost):
    # 打印训练的中间结果，训练轮次，batch数，损失函数
    print("Pass %d, Batch %d, Cost %f" % (pass_id,batch_id, cost))


use_cuda = False  # 如想使用GPU，请设置为 True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 调用train_program 获取预测值，损失值，
prediction, [avg_loss, acc] = train_program()

# 输入的原始图像数据，名称为img，大小为28*28*1
# 标签层，名称为label,对应输入图片的类别标签
# 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
feeder = fluid.DataFeeder(feed_list=['img', 'label'], place=place)

# 选择Adam优化器
optimizer = optimizer_program()
optimizer.minimize(avg_loss)


