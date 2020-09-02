#!/usr/bin/env python3

import sys, os
import matplotlib as mat
mat.use('GTK3Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans
import numpy as np
import sqlite3
from joblib import Parallel, delayed
import logging
from typing import *

CLUSTERS = 5


connection : sqlite3.Connection = None
images = Dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.DEBUG)

def process(file):
    if not file.endswith("jpg"):
        return
    global connection
    logging.debug("reading {}".format(file))
    img = cv2.imread(file)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    logging.debug("KMeans for {}".format(file))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(np.float32(img), CLUSTERS, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # kmeans = KMeans(n_clusters=CLUSTERS)
    # kmeans.fit(img)
    # center = kmeans.cluster_centers_
    # label = kmeans.labels_

    numLabels = np.arange(0, CLUSTERS + 1)
    (hist, _) = np.histogram(label, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colors = center[(-hist).argsort()]
    # c = connection.cursor()
    color_ints = []

    logging.debug("{}: {}".format(file, str(colors)))
    for i in range(CLUSTERS):
        col = int(colors[i][2]) << 16 | int(colors[i][1]) << 8 | int(colors[i][0])
        assert col <= 2**24
        color_ints.append(col)

    logging.debug("{}: {}".format(file, str(color_ints)))

    #c.execute("INSERT INTO images VALUES (\"{}\",{})".format(os.path.basename(file),",".join([str(i) for i in color_ints])))
    #connection.commit()
    #c.close()
    images.update({os.path.basename(file): color_ints})

def main(folder, img, db):
    global connection
    logging.info("Createing Database: hist_{}.sqlite".format(db))
    connection = sqlite3.connect("hist_{}.sqlite".format(db), check_same_thread = False)
    c = connection.cursor()
    c.execute("CREATE TABLE images (imgname text)")
    for i in range(CLUSTERS):
        c.execute("ALTER TABLE images ADD COLUMN color{} integer".format(i))
    connection.commit()

    for root, dirs, files in os.walk(folder):
        Parallel(n_jobs=2)(delayed(process)(os.path.join(root, file)) for file in files)
        #[process(os.path.join(root, file)) for file in files]

    for key, value in images.items():
        c.execute("INSERT INTO images VALUES (\"{}\",{})".format(key,
                                                                 ",".join([str(i) for i in value])))
    connection.commit()
    connection.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    assert len(sys.argv) > 3
    folder = sys.argv[1]
    img = sys.argv[2]
    db = sys.argv[3]
    main(folder, img, db)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
