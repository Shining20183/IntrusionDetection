# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from inputdata import DataSet
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from modifiedCell import ModifiedLSTMCell
from datetime import datetime

#modified

def data2feature(f_name, cla):
    f_value = f_name.values
    time, length, packet = np.split(f_value, [10, 20], axis=1)
    times = np.split(time, 10, axis=1)
    lengths = np.split(length, 10, axis=1)
    packets = np.split(packet, 10, axis=1)
    res = np.concatenate((times[0], lengths[0], packets[0]), axis=1)
    for i in range(1, 10):
        res = np.concatenate((res, times[i], lengths[i], packets[i]), axis=1)

    label = np.zeros(res.shape[0])
    feature = np.insert(res, 0, values=label, axis=1)
    feature[:, -1] = cla
    np.random.shuffle(feature)
    print(feature.shape)
    return feature


def discard_fiv_tupple(data):
    for i in range(10):
        data[:, 10 + i * 162] = 0
        data[:, 13 + i * 162:25 + i * 162] = 0
    return data

print('data')

benign = pd.read_csv('input/labeled_nBENIGN.csv')
print('finish benign')
Bot = pd.read_csv('input/labeled_Bot.csv')
print('finish bot')
DDoS = pd.read_csv('input/labeled_DDoS.csv')
print('finish ddos')
DoSGoldenEye = pd.read_csv('input/labeled_DoSGoldenEye.csv')
print('finish dosgoldeneye')
DoSHulk = pd.read_csv('input/labeled_DoSHulk.csv')
print('finish doshulk')
DoSSlowhttptest = pd.read_csv('input/labeled_DoSSlowhttptest.csv')
print('finish dosslowhttptest')
DoSslowloris = pd.read_csv('input/labeled_DoSslowloris.csv')
print('finish dossloworis')
FTPPatator = pd.read_csv('input/labeled_FTPPatator.csv')
print('finish ftp patator')
PortScan = pd.read_csv('input/labeled_PortScan.csv')
print('finish portscan')
SSHPatator = pd.read_csv('input/labeled_SSHPatator.csv')
print('finish ssh')
WebAttackBruteForce = pd.read_csv('input/labeled_WebAttackBruteForce.csv')
print('finish brute force')
WebAttackXSS = pd.read_csv('input/labeled_WebAttackXSS.csv')
print('finish xss')

d0 = data2feature(benign, 0)
del benign
print('0')
d1 = data2feature(Bot, 1)
del Bot
print('1')
d2 = data2feature(DDoS, 2)
del DDoS
print('2')
d3 = data2feature(DoSGoldenEye, 3)
del DoSGoldenEye
print('3')
d4 = data2feature(DoSHulk, 3)
del DoSHulk
print('4')
d5 = data2feature(DoSSlowhttptest, 3)
del DoSSlowhttptest
print('5')
d6 = data2feature(DoSslowloris, 3)
del DoSslowloris
print('6')
d7 = data2feature(FTPPatator, 4)
del FTPPatator
print('7')
d8 = data2feature(PortScan, 5)
del PortScan
print('8')
d9 = data2feature(SSHPatator, 4)
del SSHPatator
print('9')
d10 = data2feature(WebAttackBruteForce, 6)
del WebAttackBruteForce
print('10')
d11 = data2feature(WebAttackXSS, 6)
del WebAttackXSS
print('11')

Data_tupple = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)
del d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11
# Data_tupple = (d3, d4, d5)

Data = np.concatenate(Data_tupple, axis=0)
Data = discard_fiv_tupple(Data)
print('finish discard')
np.random.shuffle(Data)
print('finish shuffle')

x_raw = np.array(Data[:, :-1], dtype='float32')
y_raw = np.array(Data[:, -1], dtype='int32')
del Data

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.25, random_state=0)
totalnum = len(x_raw)
trainnum = len(data_train)
testnum = len(data_test)
del x_raw,y_raw

print('finish load data!')

# 将一维 n类的标签转化为n维的标签
def labels_transform(mlist, classes):
    # print('label_transform function.....')
    # print(mlist.shape)
    batch_label = np.zeros((len(mlist), classes), dtype='i4')
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


# ==========================================================================


# ==========================================================================


lr = 0.0001

batch_size = tf.placeholder(tf.int32, shape=[])

# input_size = 160
input_size = 162

timestep_size = 10

hidden_size = 256

layer_num = 2
class_num = 7

_X = tf.placeholder(tf.float32, [None, timestep_size * input_size])
y = tf.placeholder(tf.int32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)
print('rnn_input_0', _X.shape)
X = tf.reshape(_X, [-1, timestep_size, input_size])
print('rnn_input', X.shape)

# rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256,256]]   # 主要要修改的应该就是LSTMCell
# rnn_layers = [LSTMCell(size) for size in [256, 256]]
rnn_layers = [ModifiedLSTMCell(size, hidden_dim=256, train=1) for size in [256]]

print('rnn_layers', len(rnn_layers))

# 多层RNN
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

init_state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)
print('init_state', init_state)

# outputs是最后一层每个step的输出 states是每一层的最后那个step的输出
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=X,
                                   initial_state=init_state, dtype=tf.float32, time_major=False)
print('outputs', outputs.shape)
print('state', len(state))
print('state[0]', len(state[0]))

h_state = state[-1][1]  # 最后一层 index为1
print('h_state', h_state.shape)  # (?,256)  之前定义的hidden state有多少个，现在就有多少

W = tf.Variable(tf.truncated_normal(shape=[hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.15, dtype=tf.float32, shape=[class_num]))
# [batch_size,hidden_size]*[hidden_size,class_num] + [class_num] --> [batch_size,class_num]
logits = tf.matmul(h_state, W) + bias

# 此处的hidden state就相当于是fully connected layer的输出

predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
loss = tf.losses.mean_squared_error(y, predictions["probabilities"])
train_op = tf.train.AdamOptimizer(learning_rate=lr, ).minimize(loss)

# 在训练过程中最小化loss

correct_prediction = tf.equal(predictions["classes"], tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

TP = tf.metrics.true_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])

print("\n" + "=" * 50 + "Benign Trainging" + "=" * 50)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
_batch_size = 128
mydata_train = DataSet(data_train, label_train)
statr = time.time()

accuracys=[]
begin_time = datetime.now()
for i in range(20000):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], class_num)
    if (i + 1) % 20 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})
        accuracys.append(train_accuracy)
        print("\nthe %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})
    # gc.collect()

end_time = datetime.now()
duarion = (end_time-begin_time).seconds
print('train time:',duarion)

file=open('accuracy-LSTM7-1.txt','w')
string = str(accuracys)
string = string.strip(']')
string = string.strip('[')
file.write(string)
file.close()
print('accuracy has been stored.')

print("\ntraining finished cost time:%f" % (time.time() - statr))

print("\n" + "=" * 50 + "Benign test" + "=" * 50)
test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
preLabel = []
mlabel = []
test_batch_size = 128  # 之前这里设置的太大了，导致之后会报错
test_iter = len(data_test) // test_batch_size + 1

mydata_test = DataSet(data_test, label_test)

test_start = time.time()
print('test iter', test_iter)
for i in range(test_iter):
    batch = mydata_test.next_batch(test_batch_size)
    mlabel = mlabel + list(batch[1])
    labels = labels_transform(batch[1], class_num)

    e_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_tp, value_tp = sess.run(TP, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_fp, value_fp = sess.run(FP, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_tn, value_tn = sess.run(TN, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    tensor_fn, value_fn = sess.run(FN, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
    preLabel = preLabel + list(sess.run(predictions["classes"], feed_dict={_X: batch[0], y: labels, keep_prob: 1.0,
                                                                           batch_size: test_batch_size}))

    test_accuracy = test_accuracy + e_accuracy
    true_positives = true_positives + value_tp
    false_positives = false_positives + value_fp
    true_negatives = true_negatives + value_tn
    false_negatives = false_negatives + value_fn

print("\ntest cost time :%d" % (time.time() - test_start))
print("\n" + "=" * 50 + "Test result" + "=" * 50)
print("\n test accuracy :%f" % (test_accuracy / test_iter))
print("\n true positives :%d" % true_positives)
print("\n false positives :%d" % false_positives)
print("\n true negatives :%d" % true_negatives)
print("\n false negatives :%d" % false_negatives)
print("\n" + "=" * 50 + "  DataSet Describe  " + "=" * 50)
print("\nAll DataSet Number:%s Trainging DataSet Number:%s Test DataSet Number:%s" % (
    totalnum, trainnum, testnum))

mP = true_positives / (true_positives + false_positives)
mR = true_positives / (true_positives + false_negatives)
mF1_score = 2 * mP * mR / (mP + mR)

print("\nPrecision:%f" % mP)
print("\nRecall:%f" % mR)
print("\nF1-Score:%f" % mF1_score)
conmat = confusion_matrix(mlabel, preLabel)
print("\nConfusion Matraics:")
print(conmat)
# print(len(mlabel))

from Visualization import Visual
matrix = Visual()
matrix.cm_plot(mlabel,preLabel,['Benign','Bot','DDoS','DoS','Patator','PortScan','WebAttack'],'LSTM_7_1')
print('finish image confusion')
print('train time:',duarion)
