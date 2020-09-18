#!/usr/bin/env python3


import sys, os
#sys.path.append("/scratch/bbecker/cv/lib/python3.6/dist-packages")

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyecm

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

dirn = os.path.dirname(os.path.realpath(__file__))

array = []

for f in os.listdir(os.path.join(dirn,"tp_data")):
    if f.endswith(".sqlite"):
        print("{}\n".format(os.path.join(dirn,"tp_data",f)))
        try:
            connection = sqlite3.connect(os.path.join(dirn,"tp_data",f))
            c = connection.cursor()
            for row in c.execute("SELECT color1 FROM images"):
                r = row[0] >> 16 & 255
                g = row[0] >> 8  & 255
                b = row[0]       & 255
                array.append([r, g, b])
            connection.close()
        except: pass

num = len(array)
print(num)
factors = list(pyecm.factors(num, False, True, 10, 1))
print(factors)
fac1 = np.prod(factors[:-1])
fac2 = factors[len(factors)-1]
print("{} x {}".format(fac1, fac2))

array = np.array(array)
print(array.shape)
array = array.reshape((fac1,fac2 , 3)).astype("uint8")
print(array.shape)
#print(array)


cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
rs = ResizeWithAspectRatio(img, height=1000)
cv2.imshow("output", rs)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("hist.jpg", img)

clr = ('b','g','r')
for i,col in enumerate(clr):
    histr = cv2.calcHist([array],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    #plt.set_xlim(0,256)

plt.show()
