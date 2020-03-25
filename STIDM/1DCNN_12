import tensorflow as tf
import numpy as np
from inputdata import DataSet
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

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
d4 = data2feature(DoSHulk_packet, 4)
del DoSHulk_packet
print('4')
d5 = data2feature(DoSSlowhttptest_packet, 5)
del DoSSlowhttptest_packet
print('5')
d6 = data2feature(DoSslowloris_packet, 6)
del DoSslowloris_packet
print('6')
d7 = data2feature(FTPPatator_packet, 7)
del FTPPatator_packet
print('7')
d8 = data2feature(PortScan_packet, 8)
del PortScan_packet
print('8')
d9 = data2feature(SSHPatator_packet, 9)
del SSHPatator_packet
print('9')
d10 = data2feature(WebAttackBruteForce_packet, 10)
del WebAttackBruteForce_packet
print('10')
d11 = data2feature(WebAttackXSS_packet, 11)
del WebAttackXSS_packet
print('11')
Data_tupple = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)

del d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11

Data = np.concatenate(Data_tupple, axis=0)
del Data_tupple
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

# ------------------------------------------------------------------------------------------

# ------------------------------ model definition ------------------------------
# parameter
learning_rate = 0.0001
max_length = 1600
classes_num = 12
batch_size = tf.placeholder(tf.int32, [])
train_iter = 20000
# train_iter = 100
_batch_size = 128


# cnn network
_X = tf.placeholder(tf.float32, [None, max_length])
y = tf.placeholder(tf.int32, [None, classes_num])
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv1d(inputs, filters, kernel_size, stride):
    return tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='SAME',
                            use_bias=True)


def max_pool(inputs):
    return tf.layers.max_pooling1d(inputs=inputs, pool_size=2, strides=2, padding='SAME')


cnn_input = tf.reshape(_X, [128, 1600, 1])
print('cnn_input',cnn_input.shape)
print(type(cnn_input))


conv1 = tf.nn.relu(conv1d(cnn_input, 32, 5, 2))  # (128, 800, 32)
pool1 = max_pool(conv1)  # (128, 400, 32)

conv2 = tf.nn.relu(conv1d(pool1, 64, 3, 1))  # (128, 400, 64)
pool2 = max_pool(conv2)  # (128, 200, 64)

pool2_flat = tf.reshape(pool2, [-1, 200 * 64])

W_fc1 = weight_variable([200 * 64, 1024])
b_fc1 = bias_variable([1024])
cnn_fc1 = tf.matmul(pool2_flat, W_fc1) + b_fc1

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

mydata_train = DataSet(data_train, label_train)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

accuracys=[]
begin_time = datetime.now()
for i in range(train_iter):
    # print('iter:', i)
    batch = mydata_train.next_batch(128)  # 此处的batch是由[128,1600]和[128,]组成的tuple，batch[0]就是tuple
    labels = labels_transform(batch[1], classes_num)
    if (i + 1) % 20 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: batch[0], y: labels,
                                                       keep_prob: 1.0, batch_size: _batch_size})
        accuracys.append(train_accuracy)
        print("\nthe %dth loop,training accuracy:%f" % (i + 1, train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: labels, keep_prob: 0.5, batch_size: _batch_size})
    # gc.collect()

end_time = datetime.now()
duarion = (end_time-begin_time).seconds
print('train time:',duarion)

file=open('accuracy-1DCNN12-1.txt','w')
string = str(accuracys)
string = string.strip(']')
string = string.strip('[')
file.write(string)
file.close()
print('accuracy has been stored.')
# ------------------------------------------------------------------------------------------


# ------------------------------ test model ------------------------------
test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
# test_batch_size = 2000
preLabel = []
test_batch_size=128
mlabel = []
test_iter = len(data_test) // test_batch_size + 1

mydata_test = DataSet(data_test, label_test)
print("\n" + "=" * 50 + "Benign test" + "=" * 50)
test_start = datetime.now()
print('test_iter:',test_iter)
for i in range(test_iter):
    batch = mydata_test.next_batch(test_batch_size)
    mlabel = mlabel + list(batch[1])
    labels = labels_transform(batch[1], classes_num)
    print('batch[0].shape:',batch[0].shape)

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

print("\ntest cost time :%d" % (datetime.now() - test_start).seconds)
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
print(len(mlabel))

from Visualization import Visual
matrix = Visual()
label12=['Benign','Bot','DDoS','DoSGoldenEye','DoSHulk','DoSSlowhttptest','DoSslowloris','FTPPatator','PortScan','SSHPatator','WebAttackBruteForce','WebAttackXSS']
matrix.cm_plot(mlabel,preLabel,label12,'1DCNN_12_1')
print('finish image confusion')
print('train time:',duarion)

# ------------------------------------------------------------------------------------------
