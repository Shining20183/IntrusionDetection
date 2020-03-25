import matplotlib.pyplot as pyplot
import os

def gettime(filepath):
    time_feature=[]
    with open(filepath,'r') as f:
        max_t=0
        min_t=255
        for line in f :
            line = line.strip('\n')
            linelist = line.split(',')
            times = linelist[0:10]
            for i in range(len(times)):
                times[i]=float(times[i])
            max_t = max(max_t,max(times))
            min_t = min(min_t,min(times))
            time_feature.append(times)
    # print('min',min_t)
    # print('max',max_t)
    for i in range(len(time_feature)):
        for j in range(len(time_feature[0])):
            cur = time_feature[i][j]
            # print('cur',cur)
            time_feature[i][j] = abs(255*(cur-min_t)/(max_t-min_t))
            # print('time[i][j]',time_feature[i][j])
    return time_feature


names=['BENIGN','Bot','DDoS','DoSGoldenEye','DoSHulk','DoSSlowhttptest','DoSslowloris','FTPPatator','Heartbleed','Infiltration','PortScan','SSHPatator','WebAttackBruteForce','WebAttackSqlInjection','WebAttackXSS']
for name in names:
    filename = 'input/labeled_'+name+'.csv'
    savepath = 'image/'+name
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    time_feature = gettime(filename)
    for i in range(len(time_feature)):
        if i == 100:
            break
        print(i)
        imgarr = [time_feature[i]]
        pyplot.imshow(imgarr)
        path = savepath+'/img-'+str(i)+'.png'
        pyplot.savefig(path)

# time_feature = gettime('input/labeled_.csv')
# for i in range(len(time_feature)):
#     if i==100:
#         break
#     print(i)
#     imgarr=[time_feature[i]]
#     pyplot.imshow(imgarr)
#     path = 'DoS slowloris/img-'+str(i)+'.png'
#     pyplot.savefig(path)
