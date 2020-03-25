import os
import csv
from datetime import datetime

# 用于在没有真实数据的时候生成模拟数据

begin= datetime.now()

path='input'
filenames = os.listdir(path)

index=0

with open('data.csv','w') as f:
    writer = csv.writer(f)
    for filename in filenames:
        protocol = filename[11:14]
        filename = filename[15:-5]
        info = filename.split('_')
        packetID = info[0].replace('-', '.') + '-' + info[2].replace('-', '.') + '-' + info[1] + '-' + info[3]
        if protocol == 'TCP':
            packetID = packetID + '-6'
        elif protocol == 'UDP':
            packetID = packetID + '-17'

        packetID2=[packetID]
        if index%3!=0:
            packetID2.append('BENIGN')
        else:
            packetID2.append('ATTACK')
        index+=1
        writer.writerow(packetID2)

end=datetime.now()
print('Duration:',(end-begin).seconds)
