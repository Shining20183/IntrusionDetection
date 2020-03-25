import tensorflow as tf
import numpy as np
from inputdata import DataSet
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from modifiedCell import ModifiedLSTMCell
import time
import pandas as pd
import gc

#modified
# ------------------------------ input preparation ------------------------------
def splitdata(filepath):
    time_feature = []
    length_feature = []
    packet_vector = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip('\n')
            linelist = line.split(',')
            time_feature.append(linelist[0:10])
            length_feature.append(linelist[10:20])
            packet_vector.append(linelist[20:])
    return time_feature, length_feature, packet_vector


def combinedata(filepath):
    feature = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip('\n')
            linelist = line.split(',')
            linefeature = []
            for i in range(10):
                linefeature += [linelist[i]]
                linefeature += [linelist[10 + i]]
                linefeature += linelist[20 + i * 160:180 + i * 160]
            feature.append(linefeature)
    print('feature', len(feature[0]))
    return feature


def data2feature(f_name, cla):
    f_value = np.array(f_name)
    label = np.zeros(f_value.shape[0])
    feature = np.insert(f_value, 0, values=label, axis=1)
    feature[:, -1] = cla
    np.random.shuffle(feature)
    return feature


# 针对有时间的情况
def discard_fiv_tupple(data):
    for i in range(10):
        data[:, 10 + i * 162] = 0
        data[:, 13 + i * 162:25 + i * 162] = 0
    return data


# 将一维 n类的标签转化为n维的标签
def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype='i4')
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


begin = datetime.now()

print('load data,',begin)

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
d4 = data2feature(DoSHulk, 4)
del DoSHulk
print('4')
d5 = data2feature(DoSSlowhttptest, 5)
del DoSSlowhttptest
print('5')
d6 = data2feature(DoSslowloris, 6)
del DoSslowloris
print('6')
d7 = data2feature(FTPPatator, 7)
del FTPPatator
print('7')
d8 = data2feature(PortScan, 8)
del PortScan
print('8')
d9 = data2feature(SSHPatator, 9)
del SSHPatator
print('9')
d10 = data2feature(WebAttackBruteForce, 10)
del WebAttackBruteForce
print('10')
d11 = data2feature(WebAttackXSS, 11)
del WebAttackXSS
print('11')

Data_tupple = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)
del d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11

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


x_use = x_raw[0].reshape([1,1620])
y_use = y_raw[0]
y_use = labels_transform([y_use],3)
print('x_use:',x_use)
print(type(x_use))
print('y_use:',y_use)

del x_raw,y_raw
# ------------------------------------------------------------------------------------------


# ------------------------------ model difinition ------------------------------
# parameter
learning_rate = 0.0005
max_length = 1620
# max_length = 1600
classes_num = 12  # 类别的个数

train_iter = 20000
# train_iter = 200
_batch_size = 128

# cnn network
with tf.name_scope("input"):
    _X = tf.placeholder(tf.float32, [None, max_length],name='xs')
    y = tf.placeholder(tf.int32, [None, classes_num],name='ys')
    keep_prob = tf.placeholder(tf.float32,name='kp')
    batch_size = tf.placeholder(tf.int32, [],name='bs')



def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv1d(inputs, filters, kernel_size, stride):
    return tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='SAME',
                            use_bias=True)


def max_pool(inputs):
    return tf.layers.max_pooling1d(inputs=inputs, pool_size=2, strides=2, padding='SAME')


cnn_input = tf.reshape(_X, [128, 1620, 1])
time_feature_1, length_feature_1, cnn_input_1, time_feature_2, length_feature_2, cnn_input_2, \
time_feature_3, length_feature_3, cnn_input_3, time_feature_4, length_feature_4, cnn_input_4, \
time_feature_5, length_feature_5, cnn_input_5, time_feature_6, length_feature_6, cnn_input_6, \
time_feature_7, length_feature_7, cnn_input_7, time_feature_8, length_feature_8, cnn_input_8, \
time_feature_9, length_feature_9, cnn_input_9, time_feature_10, length_feature_10, cnn_input_10 = tf.split(
    cnn_input,
    [1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160, 1, 1, 160], 1)

time_feature_1 = tf.reshape(time_feature_1, [batch_size, 1])
time_feature_2 = tf.reshape(time_feature_2, [batch_size, 1])
time_feature_3 = tf.reshape(time_feature_3, [batch_size, 1])
time_feature_4 = tf.reshape(time_feature_4, [batch_size, 1])
time_feature_5 = tf.reshape(time_feature_5, [batch_size, 1])
time_feature_6 = tf.reshape(time_feature_6, [batch_size, 1])
time_feature_7 = tf.reshape(time_feature_7, [batch_size, 1])
time_feature_8 = tf.reshape(time_feature_8, [batch_size, 1])
time_feature_9 = tf.reshape(time_feature_9, [batch_size, 1])
time_feature_10 = tf.reshape(time_feature_10, [batch_size, 1])

length_feature_1 = tf.reshape(length_feature_1, [batch_size, 1])
length_feature_2 = tf.reshape(length_feature_2, [batch_size, 1])
length_feature_3 = tf.reshape(length_feature_3, [batch_size, 1])
length_feature_4 = tf.reshape(length_feature_4, [batch_size, 1])
length_feature_5 = tf.reshape(length_feature_5, [batch_size, 1])
length_feature_6 = tf.reshape(length_feature_6, [batch_size, 1])
length_feature_7 = tf.reshape(length_feature_7, [batch_size, 1])
length_feature_8 = tf.reshape(length_feature_8, [batch_size, 1])
length_feature_9 = tf.reshape(length_feature_9, [batch_size, 1])
length_feature_10 = tf.reshape(length_feature_10, [batch_size, 1])

cnn_input = tf.concat([cnn_input_1, cnn_input_2, cnn_input_3, cnn_input_4, cnn_input_5, cnn_input_6, cnn_input_7,
                       cnn_input_8, cnn_input_9, cnn_input_10], 1)

conv1 = tf.nn.relu(conv1d(cnn_input, 32, 5, 2))  # (128, 800, 32)
pool1 = max_pool(conv1)  # (128, 400, 32)

conv2 = tf.nn.relu(conv1d(pool1, 64, 3, 1))  # (128, 400, 64)
pool2 = max_pool(conv2)  # (128, 200, 64)

pool2_flat = tf.reshape(pool2, [-1, 200 * 64])

W_fc1 = weight_variable([200 * 64, 1600])
b_fc1 = bias_variable([1600])
cnn_fc1 = tf.matmul(pool2_flat, W_fc1) + b_fc1

cnn_fc1_drop = tf.nn.dropout(cnn_fc1, keep_prob)

# LSTM network


lr = 0.0001

# batch_size = tf.placeholder(tf.int32, shape=[])
# batch_size = tf.placeholder(tf.int32,[])

# input_size = 160
lstm_input_size = 162

lstm_timestep_size = 10

hidden_size = 256

layer_num = 2
class_num = 12

lstm_input_1, lstm_input_2, lstm_input_3, lstm_input_4, lstm_input_5, lstm_input_6, lstm_input_7, lstm_input_8, lstm_input_9, lstm_input_10 = tf.split(
    cnn_fc1_drop, num_or_size_splits=10, axis=1)
lstm_input_t = tf.concat(
    [time_feature_1, length_feature_1, lstm_input_2, time_feature_2, length_feature_2, lstm_input_2, time_feature_3,
     length_feature_3, lstm_input_3, time_feature_4, length_feature_4, lstm_input_4, time_feature_5, length_feature_5,
     lstm_input_5, time_feature_6, length_feature_6, lstm_input_6, time_feature_7, length_feature_7, lstm_input_7,
     time_feature_8, length_feature_8, lstm_input_8, time_feature_9, length_feature_9, lstm_input_9, time_feature_10,
     length_feature_10, lstm_input_10], 1)

lstm_input = tf.reshape(lstm_input_t, [-1, lstm_timestep_size, lstm_input_size])

# _X = tf.placeholder(tf.float32, [None, lstm_timestep_size * lstm_input_size])
# y = tf.placeholder(tf.int32, [None, class_num])
# keep_prob = tf.placeholder(tf.float32)

X = tf.reshape(_X, [-1, lstm_timestep_size, lstm_input_size])
# X = tf.reshape(lstm_input, [-1, lstm_timestep_size, lstm_input_size])

# 以下是对之前部分的修改
# X = lstm_input
# y=y


# rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256,256]]   # 主要要修改的应该就是LSTMCell
# rnn_layers = [LSTMCell(size) for size in [256, 256]]
rnn_layers = [ModifiedLSTMCell(size, hidden_dim=256, train=1) for size in [256]]

# 多层RNN
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

init_state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)

# outputs是最后一层每个step的输出 states是每一层的最后那个step的输出
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=X,
                                   initial_state=init_state, dtype=tf.float32, time_major=False)
h_state = state[-1][1]  # 最后一层 index为1

W = tf.Variable(tf.truncated_normal(shape=[hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.15, dtype=tf.float32, shape=[class_num]))
# [batch_size,hidden_size]*[hidden_size,class_num] + [class_num] --> [batch_size,class_num]
logits = tf.matmul(h_state, W) + bias

# 此处的hidden state就相当于是fully connected layer的输出

predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}

with tf.name_scope('output'):
    predictionResult = tf.argmax(input=logits, axis=1,name="pre")

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
loss = tf.losses.mean_squared_error(y, predictions["probabilities"])
train_op = tf.train.AdamOptimizer(learning_rate=lr, ).minimize(loss)

correct_prediction = tf.equal(predictions["classes"], tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

TP = tf.metrics.true_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])

# ------------------------------------------------------------------------------------------

# ------------------------------ train model ------------------------------
print("\n" + "=" * 50 + "Benign Trainging" + "=" * 50)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# sess = tf.Session(config=config)

##### errors_impl.InternalError: Failed to create session. 看看这里为什么报错

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
_batch_size = 128

mydata_train = DataSet(data_train, label_train)


accuracys=[]
begin_time = datetime.now()

for i in range(train_iter):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], class_num)
    if (i + 1) % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})
        print("\nthe %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
        accuracys.append(train_accuracy)
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})
    # gc.collect()

end_time = datetime.now()
duarion = (end_time-begin_time).seconds
print('train time:',duarion)

# # save model method 2
saver = tf.train.Saver()
saver.save(sess, './model/classifier.ckpt')

print('model saved.')

# ------------------------------------------------------------------------------------------


# ------------------------------ test model ------------------------------
# print("\n" + "=" * 50 + "Benign test" + "=" * 50)
# test_accuracy = 0
# true_positives = 0
# false_positives = 0
# true_negatives = 0
# false_negatives = 0
# preLabel = []
# mlabel = []
# test_batch_size = 128  # 之前这里设置的太大了，导致之后会报错
# test_iter = len(data_test) // test_batch_size + 1
#
# mydata_test = DataSet(data_test, label_test)
#
# test_start = time.time()
# print('test iter', test_iter)
# for i in range(test_iter):
#     batch = mydata_test.next_batch(test_batch_size)
#     mlabel = mlabel + list(batch[1])
#     labels = labels_transform(batch[1], class_num)
#
#     e_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
#     tensor_tp, value_tp = sess.run(TP, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
#     tensor_fp, value_fp = sess.run(FP, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
#     tensor_tn, value_tn = sess.run(TN, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
#     tensor_fn, value_fn = sess.run(FN, feed_dict={_X: batch[0], y: labels, keep_prob: 1.0, batch_size: test_batch_size})
#     preLabel = preLabel + list(sess.run(predictions["classes"], feed_dict={_X: batch[0], y: labels, keep_prob: 1.0,
#                                                                            batch_size: test_batch_size}))
#
#     test_accuracy = test_accuracy + e_accuracy
#     true_positives = true_positives + value_tp
#     false_positives = false_positives + value_fp
#     true_negatives = true_negatives + value_tn
#     false_negatives = false_negatives + value_fn
#
# print("\ntest cost time :%d" % (time.time() - test_start))
# print("\n" + "=" * 50 + "Test result" + "=" * 50)
# print("\n test accuracy :%f" % (test_accuracy / test_iter))
# print("\n true positives :%d" % true_positives)
# print("\n false positives :%d" % false_positives)
# print("\n true negatives :%d" % true_negatives)
# print("\n false negatives :%d" % false_negatives)
# print("\n" + "=" * 50 + "  DataSet Describe  " + "=" * 50)
# print("\nAll DataSet Number:%s Trainging DataSet Number:%s Test DataSet Number:%s" % (
#     totalnum, trainnum, testnum))
#
# mP = true_positives / (true_positives + false_positives)
# mR = true_positives / (true_positives + false_negatives)
# mF1_score = 2 * mP * mR / (mP + mR)
#
# print("\nPrecision:%f" % mP)
# print("\nRecall:%f" % mR)
# print("\nF1-Score:%f" % mF1_score)
# conmat = confusion_matrix(mlabel, preLabel)
# print("\nConfusion Matraics:")
# print(conmat)
# print(len(mlabel))
#
# from Visualization import Visual
# matrix = Visual()
# label12=['Benign','Bot','DDoS','DoSGoldenEye','DoSHulk','DoSSlowhttptest','DoSslowloris','FTPPatator','PortScan','SSHPatator','WebAttackBruteForce','WebAttackXSS']
# matrix.cm_plot(mlabel,preLabel,label12,'1DCNN_LSTM_12_1')
# print('finish image confusion')
# print('train time:',duarion)
# ------------------------------------------------------------------------------------------


# --------------------------use model ---------------------------------
#
# sess2 = tf.Session()  # 初始化session
# # 加载模型
# sess2.run(tf.global_variables_initializer())
# sess2.run(tf.local_variables_initializer())
# saver = tf.train.import_meta_graph('./model/classifier.ckpt.meta')  # 先加载meta文件，具体到文件名
# saver.restore(sess2, tf.train.latest_checkpoint('./model'))  # 加载检查点文件checkpoint，具体到文件夹即可
# graph = tf.get_default_graph()  # 绘制tensorflow图
#
# xs_u = graph.get_tensor_by_name('input/xs:0')  # 获取占位符xs
# ys_u = graph.get_tensor_by_name('input/ys:0')  # 获取占位符ys
# keep_prob = graph.get_tensor_by_name('input/kp:0')
# batch_size = graph.get_tensor_by_name('input/bs:0')
#
#
# output=sess2.graph.get_tensor_by_name('output/pre:0')
#
# classes=sess2.run(output,feed_dict={xs_u:x_use,ys_u:y_use,keep_prob: 1.0, batch_size: 1})
# print(classes)
# ------------------------------------------------------------------------------------------