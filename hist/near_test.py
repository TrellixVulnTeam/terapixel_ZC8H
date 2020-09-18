#!/usr/bin/env python3


import sys, os
#sys.path.append("/scratch/bbecker/cv/lib/python3.6/dist-packages")

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors

dirn = os.path.dirname(os.path.realpath(__file__))

array = []

def arrpack(arr):
    return (arr[0] & 255) << 16 | (arr[1] & 255) << 8 | (arr[2] & 255)

for f in os.listdir(os.path.join(dirn,"tp_data")):
    if f.endswith(".sqlite"):
        #print("{}\n".format(os.path.join(dirn,"tp_data",f)))
        connection = sqlite3.connect(os.path.join(dirn,"tp_data",f))
        c = connection.cursor()
        for row in c.execute("SELECT color1 FROM images"):
            r = row[0] >> 16 & 255
            g = row[0] >> 8  & 255
            b = row[0]       & 255
            array.append([r, g, b])
        connection.close()


array = np.array(array)


img = cv2.cvtColor(cv2.imread("0179.jpg"),cv2.COLOR_BGR2RGB)

old_shape = img.shape

img = img.reshape((old_shape[0] * old_shape[1], old_shape[2]))
print(img)

print("np.unique(img)")
search, inverse, counts = np.unique(img, return_counts=True, return_inverse=True, axis=0)

print("{} pixels, {} candidates".format(len(img), len(array)))
print("search =")
print(search)
print("counts =")
print(counts)
print("inverse =")
print(inverse)

#counts_bw = np.array([counts[0], counts[-1]])
#counts_clr = counts[1:-2]


#print("counts_bw =")
#print(counts_bw)

#print("counts_clr =")
#print(counts_clr)

print("fit(array) clr/{}".format(counts.max()))
nn_clr = NearestNeighbors(n_neighbors=counts.max(), algorithm='auto', n_jobs=-1).fit(array)
print(nn_clr)
print("kneighbors(search)")
distances, indices = nn_clr.kneighbors(search)

print("indices = ")
print(indices)
# ^- indeces of array for each search_clr



sys.exit(0)







search_pack = [arrpack(p) for p in search]
img_pack = [arrpack(p) for p in img]


    

img2 = [array[i[0]] for i in indices]

img2.reshape(old_shape)

cv2.imshow("", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
