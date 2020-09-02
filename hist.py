#!/usr/bin/env python3


import sys, os
#sys.path.append("/scratch/bbecker/cv/lib/python3.6/dist-packages")

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2


dirn = os.path.dirname(os.path.realpath(__file__))

array = []

for f in os.listdir(dirn):
    if f.endswith(".sqlite"):
        print("{}\n".format(f))
        connection = sqlite3.connect(f)
        c = connection.cursor()
        for row in c.execute("SELECT color1 FROM images"):
            r = row[0] >> 16 & 255
            g = row[0] >> 8  & 255
            b = row[0]       & 255
            array.append([r, g, b])
        connection.close()

array = np.array(array[:2351622])
print(array.shape)
array = array.reshape((1534,1533 , 3)).astype("uint8")
print(array.shape)
#print(array)


#img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

cv2.imshow("", array)
cv2.waitKey(0)
clr = ('b','g','r')
for i,col in enumerate(clr):
    histr = cv2.calcHist([array],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    #plt.set_xlim(0,256)

plt.show()