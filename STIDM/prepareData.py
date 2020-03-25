import os
import dpkt
import socket
import binascii
import numpy as np
import csv
from datetime import datetime

# 从已经分好类的flow中提取有用的信息

def getInfo(filename):
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
    return time_feature,len_feature,packet_vector

# 将16进制表示的字符串转化为10进制list
def hex2int(string):
    res=[]
    for i in range(len(string)//2):
        res.append(int(string[2*i:2*i+2],16))
    return res





# path='BENIGN'
foldernames=['DoSGoldenEye','DoSHulk','DoSSlowhttptest','DoSslowloris','FTPPatator','Heartbleed','Infiltration','PortScan','SSHPatator','WebAttackBruteForce','WebAttackSqlInjection','WebAttackXSS']
for foldername in foldernames:
    begin = datetime.now()
    print('========================'+foldername+'===========================')
    print('begin:', begin)
    path = './../data/' + foldername
    files = os.listdir(path)

    countline = 0

    with open('labeled_' + foldername + '.csv', 'w') as f:
        writer = csv.writer(f)
        for file in files:
            filename = path + '/' + file
            time_feature, len_feature, packet_vector = getInfo(filename)
            packet_vector2 = []
            for item in packet_vector:
                packet_vector2 += hex2int(item)
            feature = time_feature + len_feature + packet_vector2
            writer.writerow(feature)
            countline += 1

    end = datetime.now()
    print('end:', end)
    print('count', countline)
    print('done!')
    print('duration:', (end - begin).seconds)
