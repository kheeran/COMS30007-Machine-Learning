import numpy as np
import math

PX = np.array([1/16,1/16,1/4,1/4,1/4,1/16,1/16])
QX = np.array([2/6, 1/16, 3/16, 1/16, 1/8, 1/16, 1/6])
'''
PX = np.array([1/2,1/2])
QX = np.array([1/4,3/4])
'''

def KBdiv(pX, qX):
    sum = 0
    data = []
    for n in range(len(pX)):
        frac = pX[n]/qX[n]
        weighted = pX[n]*math.log(frac, 2)
        data.append(weighted)
        sum += weighted
    return sum, data

A,B = KBdiv(PX,QX)
An, Bn = KBdiv(QX,PX)
print (str(PX) + " sum = " + str(np.sum(PX)))
print (str(QX) + " sum = " + str(np.sum(QX)))
print (str(B) + " sum = " + str(A))
print (str(Bn) + " sum = " + str(An))
