import csv

with open('input/labeled_BENIGN.csv','r') as fout:
    with open('input/labeled_nBENIGN.csv','w') as fin:
        writer = csv.writer(fin)
        count = 0
        for line in fout:
            count+=1
            if count%5==0:
                line = line.strip('\n')
                line = line.split(',')
                writer.writerow(line)
                print(count)