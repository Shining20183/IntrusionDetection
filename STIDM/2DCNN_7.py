import tensorflow as tf
import numpy as np
from inputdata import DataSet
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import time

# modified
# ------------------------------ input preparation ------------------------------
def data2feature(f_name, cla):
    label = np.zeros(f_name.shape[0])
    feature = np.insert(f_name, 0, values=label, axis=1)
    feature[:, -1] = cla
    np.random.shuffle(feature)
    return feature


def discard_fiv_tupple(data):
    for i in range(10):
        data[:, 8 + i * 160] = 0
        data[:, 11 + i * 160:23 + i * 160] = 0
    return data


def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype='i4')
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label


begin = datetime.now()
print('begin', begin)

print('load data......')
#
benign_packet = pd.read_csv('input/labeled_nBENIGN.csv').values[:, 20:]
print('benign finish')
Bot_packet = pd.read_csv('input/labeled_Bot.csv').values[:, 20:]
print('bot finsh')
DDoS_packet = pd.read_csv('input/labeled_DDoS.csv').values[:, 20:]
print('DDOS finish')
DoSGoldenEye_packet = pd.read_csv('input/labeled_DoSGoldenEye.csv').values[:, 20:]
print('GoldenEye finish')
DoSHulk_packet = pd.read_csv('input/labeled_DoSHulk.csv').values[:, 20:]
print('DoSHulk finish')
DoSSlowhttptest_packet = pd.read_csv('input/labeled_DoSSlowhttptest.csv').values[:, 20:]
print('DoSSlowhttptest finish')
DoSslowloris_packet = pd.read_csv('input/labeled_DoSslowloris.csv').values[:, 20:]
print('DoSslowloris finish')
FTPPatator_packet = pd.read_csv('input/labeled_FTPPatator.csv').values[:, 20:]
print('FTPPatator finish')
PortScan_packet = pd.read_csv('input/labeled_PortScan.csv').values[:, 20:]
print('portscan finish')
SSHPatator_packet = pd.read_csv('input/labeled_SSHPatator.csv').values[:, 20:]
print('ssh finish')
WebAttackBruteForce_packet = pd.read_csv('input/labeled_WebAttackBruteForce.csv').values[:, 20:]
print('web brute finish')
WebAttackXSS_packet = pd.read_csv('input/labeled_WebAttackXSS.csv').values[:, 20:]
print('xss finish')

d0 = data2feature(benign_packet, 0)
del benign_packet
print('0')
d1 = data2feature(Bot_packet, 1)
del Bot_packet
print('1')
d2 = data2feature(DDoS_packet, 2)
del DDoS_packet
print('2')
d3 = data2feature(DoSGoldenEye_packet, 3)
del DoSGoldenEye_packet
print('3')
d4 = data2feature(DoSHulk_packet, 3)
del DoSHulk_packet
print('4')
d5 = data2feature(DoSSlowhttptest_packet, 3)
del DoSSlowhttptest_packet
print('5')
d6 = data2feature(DoSslowloris_packet, 3)
del DoSslowloris_packet
print('6')
d7 = data2feature(FTPPatator_packet, 4)
del FTPPatator_packet
print('7')
d8 = data2feature(PortScan_packet, 5)
del PortScan_packet
print('8')
d9 = data2feature(SSHPatator_packet, 4)
del SSHPatator_packet
print('9')
d10 = data2feature(WebAttackBruteForce_packet, 6)
del WebAttackBruteForce_packet
print('10')
d11 = data2feature(WebAttackXSS_packet, 6)
del WebAttackXSS_packet
print('11')

# # code for test
# d3 = data2feature(DoSGoldenEye_packet, 0)
# del DoSGoldenEye_packet
# print('3')
# d4 = data2feature(DoSHulk_packet, 1)
# del DoSHulk_packet
# print('4')
# d5 = data2feature(DoSSlowhttptest_packet, 2)
# del DoSSlowhttptest_packet
# print('5')
# Data_tupple = (d3, d4, d5)
# del d3, d4, d5

Data_tupple = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)
del d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11

Data = np.concatenate(Data_tupple, axis=0)
Data = discard_fiv_tupple(Data)
print('diacard finish')
np.random.shuffle(Data)
print('shuffle finish')

x_raw = np.array(Data[:, :-1], dtype='float32')
y_raw = np.array(Data[:, -1], dtype='int32')

del Data

data_train, data_test, label_train, label_test = train_test_split(x_raw, y_raw, test_size=0.25, random_state=0)
totalnum = len(x_raw)
trainnum = len(data_train)
testnum = len(data_test)
del x_raw, y_raw
print('finish load data!')

# ---------------------- model define ------------------------

# parameter
learning_rate = 0.0005
img_shape = 40 * 40
classes_num = 7
batch_size = tf.placeholder(tf.int32, [])
lstm_input_size = 160
lstm_timestep_size = 10
lstm_hidden_layers = 2
train_iter = 20000

# cnn network
_X = tf.placeholder(tf.float32, [None, img_shape])
y = tf.placeholder(tf.int32, [None, classes_num])
keep_prob = tf.placeholder(tf.float32)


# 生成成正态分布的weight
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


cnn_input = tf.reshape(_X, [-1, 40, 40, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv_1 = tf.nn.relu(conv2d(cnn_input, W_conv1) + b_conv1)  # (128, 36, 36, 32)

pool_1 = max_pool(conv_1)  # (128, 18, 18, 32)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
conv_2 = tf.nn.relu(conv2d(pool_1, W_conv2) + b_conv2)  # (128, 16, 16, 64)
# 16*16*64
pool_2 = max_pool(conv_2)  # (128, 8, 8, 64)
# 8*8*64 = 4096

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
pool_2_flat = tf.reshape(pool_2, [-1, 8 * 8 * 64])  # (128, 4096)
cnn_fc1 = tf.matmul(pool_2_flat, W_fc1) + b_fc1
cnn_fc1_drop = tf.nn.dropout(cnn_fc1, keep_prob)

W_fc2 = weight_variable([1024, classes_num])
b_fc2 = bias_variable([classes_num])
logits = tf.matmul(cnn_fc1_drop, W_fc2) + b_fc2

predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}
y = tf.one_hot(indices=tf.argmax(input=y, axis=1), depth=classes_num, dtype="int32")
loss = tf.losses.softmax_cross_entropy(y, logits)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, ).minimize(loss)

correct_prediction = tf.equal(predictions["classes"], tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

TP = tf.metrics.true_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=predictions["classes"])

sess = tf.Session()
print("\n" + "=" * 50 + "Benign Trainging" + "=" * 50)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())  # 初始化局部变量
_batch_size = 128
mydata_train = DataSet(data_train, label_train)

start = time.time()

accuracys=[]
begin_time = datetime.now()
for i in range(train_iter):
    batch = mydata_train.next_batch(_batch_size)
    labels = labels_transform(batch[1], classes_num)
    if (i + 1) % 20 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})
        accuracys.append(train_accuracy)
        print("\nthe %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5,
                                  batch_size: _batch_size})

end_time = datetime.now()
duarion = (end_time-begin_time).seconds
print('train time:',duarion)

file=open('accuracy-2DCNN7-1.txt','w')
string = str(accuracys)
string = string.strip(']')
string = string.strip('[')
file.write(string)
file.close()
print('accuracy has been stored.')

print("\ntraining finished cost time:%f" % (time.time() - start))

test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
test_batch_size = 128
preLabel = []
mlabel = []
test_iter = len(data_test) // test_batch_size + 1

mydata_test = DataSet(data_test, label_test)
print("\n" + "=" * 50 + "Benign test" + "=" * 50)
test_start = time.time()
for i in range(test_iter):
    batch = mydata_test.next_batch(test_batch_size)
    mlabel = mlabel + list(batch[1])
    labels = labels_transform(batch[1], classes_num)

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
matrix.cm_plot(mlabel,preLabel,['Benign','Bot','DDoS','DoS','Patator','PortScan','WebAttack'],'2DCNN_7_1')
print('finish image confusion')
print('train time:',duarion)
