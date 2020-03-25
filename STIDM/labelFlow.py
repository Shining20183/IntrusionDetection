import os,shutil
from datetime import datetime
import re

# 根据标签文件pcap文件放置到对应类型的文件夹中

movecount = 0

def movefile(srcdir,dstdir):
    global movecount
    if not os.path.isfile(srcdir):
        print('========== %s not exist!' %srcdir)
    else:
        if not os.path.exists(dstdir):
            print('create new dir ',dstdir)
            os.makedirs(dstdir)
        shutil.move(srcdir,dstdir)
        movecount+=1
        print('########## move %s' %srcdir)

begin = datetime.now()
with open('./../../dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv','r',encoding='ISO-8859-1') as f:  #,encoding='ISO-8859-1'
    passed=0
    count=0
    for line in f:
        count+=1
        print(count)
        if count==1:
            continue
        line = line.strip('\n')
        linelist = line.split(',')
        info=linelist[0]
        label=linelist[-1]
        label=re.sub('[^a-zA-Z]','',label)
        print('label:',label)
        infolist = info.split('-')
        if(len(infolist)!=5):
            print('?????????? continue')
            continue
        print(infolist)
        src = infolist[0].replace('.','-')
        dst = infolist[1].replace('.','-')
        sport = infolist[2]
        dport = infolist[3]
        prop = infolist[4]

        filename='Friday-WorkingHours-fixed.pcap.'
        if prop == '6':
            filename =filename+ 'TCP_'
        elif prop == '17':
            filename = filename + 'UDP_'
        else:
            passed+=1
            print('---------- pass')
        filename = filename+src+'_'+sport+'_'+dst+'_'+dport+'.pcap'

        srcfilepath = './../../dataset/Friday-WorkingHours-fixed/'+filename

        movefile(srcfilepath,'./../'+label)

end = datetime.now()
print('duration:',(end-begin).seconds)
print('passed:',passed)
print('moved:',movecount)
print('count:',count)


# Wednesday-WorkingHours-fixed.pcap.TCP_172-16-0-1_37646_192-168-10-50_80.pcap
