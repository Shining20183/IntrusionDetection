import os
import dpkt
import socket
import binascii
import numpy as np
import csv
from datetime import datetime
import time
import datetime
import tensorflow as tf

# 将16进制表示的字符串转化为10进制list
def hex2int(string):
    res=[]
    for i in range(len(string)//2):
        res.append(int(string[2*i:2*i+2],16))
    return res

# 处理上传的文件
def getFeature(filename):
    time_feature=[]
    len_feature=[]
    packet_vector=[]
    f = open(filename,'rb')
    pcap = dpkt.pcap.Reader(f)
    index = -1
    ts0 = 0
    for ts, buf in pcap:
        # print('index',index)
        index=index+1

        eth = dpkt.ethernet.Ethernet(buf)
        if not isinstance(eth.data, dpkt.ip.IP):   #不处理IPV6数据包
            continue

        # time feature
        if index == 0:
            # print('append 0')
            ts0 = ts
            time_feature.append('0')
        else:
            # print('----------')
            ts = ts - ts0
            time_feature.append(str(ts))

        # len feature
        len_feature.append(str(len(buf)))

        # packet feature
        packet0 = str(binascii.hexlify(buf))[:-1]
        packet = packet0[34:354]  #194
        # print(packet)
        # print(type(packet))
        # print(len(packet))
        # padding 0
        if len(packet)<320:
            for i in range(320-len(packet)):
                packet+='0'
        packet_vector.append(packet)

        if(len(time_feature)==10):
            break

    #如果不足16个包
    if len(time_feature)<10:
        zeros=''
        for j in range(320):
            zeros+='0'
        for i in range(10-len(time_feature)):
            time_feature.append('0')
            len_feature.append('0')
            packet_vector.append(zeros)
    packet_vector2 = []
    for item in packet_vector:
        packet_vector2 += hex2int(item)
    # print('time_feature:',time_feature)
    # print('len_feature:',len_feature)
    # print('packet:',packet_vector2)
    feature = time_feature + len_feature + packet_vector2
    return feature

# 将数据转化为适合模型输入的格式
def data2feature(f_name, cla):
    f_value = np.array(f_name)
    label = np.zeros(f_value.shape[0])
    feature = np.insert(f_value, 0, values=label, axis=1)
    feature[:, -1] = cla
    np.random.shuffle(feature)
    return feature

def labels_transform(mlist, classes):
    batch_label = np.zeros((len(mlist), classes), dtype='i4')
    for i in range(len(mlist)):
        batch_label[i][mlist[i]] = 1
    return batch_label




# 对流量进行分类 待补充
def classify(feature):
    x_use = np.array(feature)
    x_use = x_use.reshape([1, 1620])
    y_use = labels_transform([0], 12)

    sess2 = tf.Session()  # 初始化session
    # 加载模型
    sess2.run(tf.global_variables_initializer())
    sess2.run(tf.local_variables_initializer())
    saver = tf.train.import_meta_graph('./static/model/classifier.ckpt.meta')  # 先加载meta文件，具体到文件名
    saver.restore(sess2, tf.train.latest_checkpoint('./static/model'))  # 加载检查点文件checkpoint，具体到文件夹即可
    graph = tf.get_default_graph()  # 绘制tensorflow图

    xs_u = graph.get_tensor_by_name('input/xs:0')  # 获取占位符xs
    ys_u = graph.get_tensor_by_name('input/ys:0')  # 获取占位符ys
    keep_prob = graph.get_tensor_by_name('input/kp:0')
    batch_size = graph.get_tensor_by_name('input/bs:0')

    output = sess2.graph.get_tensor_by_name('output/pre:0')

    classes = sess2.run(output, feed_dict={xs_u: x_use, ys_u: y_use, keep_prob: 1.0, batch_size: 1})

    pattern = {0:"benign",1:"Bot",2:"DDoS",3:"DoSGoldenEye",4:"DoSHulk",5:"DoSSlowhttptest",6:"DoSslowloris",
               7:"FTPPatator",8:"PortScan",9:"SSHPatator",10:"WebAttackBruteForce",11:"WebAttackXSS"}
    return pattern[classes[0]]


def inet_to_str(inet):
    try:
        return socket.inet_ntop(socket.AF_INET,inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6,inet)


def getBasicInfo(filename):
    f = open(filename, 'rb')
    pcap = dpkt.pcap.Reader(f)
    index=-1
    begints=0
    endts=0
    detail=[]
    for ts,buf in pcap:
        print(buf)
        index=index+1
        eth = dpkt.ethernet.Ethernet(buf)
        print(eth)
        print(str(binascii.hexlify(buf)))
        print('type:',type(eth))
        if not isinstance(eth.data, dpkt.ip.IP):  # 不处理IPV6数据包
            continue
        detail.append(eth)
        if index==0:
            begints=ts
            ip=eth.data
            tcp=ip.data
            IP1 = inet_to_str(ip.src)
            IP2 = inet_to_str(ip.dst)
            port1 = tcp.sport
            port2 = tcp.dport
        endts=ts
        dur=(endts-begints)
    return index+1,IP1,IP2,port1,port2,dur

def getDetailInfo(filename):
    f = open(filename, 'rb')
    pcap = dpkt.pcap.Reader(f)
    details=[]
    index=0
    for ts, buf in pcap:
        detail={}
        eth = dpkt.ethernet.Ethernet(buf)
        if not isinstance(eth.data, dpkt.ip.IP):  # 不处理IPV6数据包
            continue
        detail['info']=eth.data
        ip=eth.data
        detail['len']=ip.len
        detail['src']=inet_to_str(ip.src)
        detail['dst']=inet_to_str(ip.dst)
        if isinstance(ip.data, dpkt.tcp.TCP):
            detail['protocol']='TCP'
        elif isinstance(ip.data, dpkt.udp.UDP):
            detail['protocol']='UDP'
        detail['ts']=ts
        index=index+1
        detail['no']=index
        details.append(detail)
    return details




def handle_uploaded_file(f):
    message={}
    try:
        if not f.name.endswith('.pcap'):
            message['status']=False
            message['failInfo']='The uploaded file is not pcap file.'
        else:
            with open('tmp.pcap', 'wb+') as des:
                for chunk in f.chunks():
                    des.write(chunk)
            feature = getFeature('tmp.pcap')
            number,IP1,IP2,port1,port2,dur = getBasicInfo('tmp.pcap')
            detail = getDetailInfo('tmp.pcap')
            message['status']=True
            message['size'] = f.size
            message['feature']=feature
            message['class']=classify(feature)
            message['number']=number
            message['IP1']=IP1
            message['IP2']=IP2
            message['dur']=dur
            message['port1']=port1
            message['port2']=port2
            message['detail']=detail
    except Exception as e:
        message['status']=False
        message['failInfo']=e
    finally:
        return message