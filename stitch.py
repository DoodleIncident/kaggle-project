import csv

with open('out/kinds.csv') as k,\
     open('out/sentiments.csv') as s,\
     open('out/whens.csv') as w,\
     open('out/ids.csv') as i:

    l0 = list(csv.reader(i, delimiter=','))
    l1 = list(csv.reader(s, delimiter=','))
    l2 = list(csv.reader(w, delimiter=','))
    l3 = list(csv.reader(k, delimiter=','))

    zusammen = [a+b+c+d for a,b,c,d in zip(l0,l1,l2,l3)]

    stiched = ["id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15".split(',')] + zusammen

    r = open('out/output.csv', 'wb')
    wr = csv.writer(r)
    wr.writerows(stiched)
