#!/usr/bin/env python3

import sys, os
#sys.path.append("/scratch/bbecker/cv/lib/python3.6/dist-packages")
#sys.path.append("/scratch/bbecker/kmcuda/src/build")
#sys.path.append("/scratch/bbecker/python-libs/lib/python3.6/site-packages/")
import cv2
from pai4sk.cluster import KMeans
import cudf, cuml
import pandas as pd
#from libKMCUDA import kmeans_cuda
#from sklearn.cluster import KMeans
import numpy as np
import sqlite3
from joblib import Parallel, delayed
import logging
from typing import *
import contextlib
from numba import cuda 
from numba.cuda.cudadrv.driver import CudaAPIError
import multiprocessing
import cupy as cp
import shutil
from datetime import datetime
from datetime import timedelta
import time

CLUSTERS = 5

DEBUG = False


connection : sqlite3.Connection = None
images = Dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', level=logging.DEBUG)

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

@contextlib.contextmanager
def nostderr():
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr

def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c,column in enumerate(df):
      pdf[str(c)] = df[column]
    return pdf

def process(filen):
    try:
        if not filen.endswith("jpg") or os.path.exists("{}.txt".format(filen)):
            return
    
        #logging.debug("reading {}".format(file))
        img = cv2.imread(filen)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
    
        #logging.debug("KMeans for {}".format(file))
    
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #ret, label, center = cv2.kmeans(np.float32(img), CLUSTERS, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        #center, label = kmeans_cuda(np.float32(img), CLUSTERS, device=1)
        worker = multiprocessing.current_process()._identity[0]
        try:
            if worker%13 > 3:
                raise RuntimeError()
            logging.info("Starting CUDA-KMeans in Worker {} on Card {} for file {}".format(worker, worker%2, os.path.basename(filen)))
            cuda.select_device(worker%2)
            kmeans = KMeans(n_clusters=CLUSTERS, n_init=5, verbose=0)
            #b, g, r = np.hsplit(img, 3)
            img_cuda = np2cudf(np.float32(img))
            with nostdout(), nostderr():
                kmeans.fit(img_cuda)
            center = cp.asarray(kmeans.cluster_centers_.values)
            label = cp.asarray(kmeans.labels_.data.mem)
    
            numLabels = cp.arange(0, CLUSTERS + 1)
            (hist, he) = cp.histogram(label, bins=numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()
            colors = (center[(-hist).argsort()]).get()
    
            del kmeans
            del img_cuda
            del center
            del label
            del numLabels
            del hist
            del he
            #cuda.close()
        except (RuntimeError, CudaAPIError):
            logging.info("Starting SKLearn-KMeans in Worker {} on CPU for file {}".format(worker, os.path.basename(filen)))
            kmeans = KMeans(n_clusters=CLUSTERS, n_init=5, precompute_distances=True, n_jobs=1, verbose=0)
            kmeans.use_gpu = False
            with nostdout(), nostderr():
                kmeans.fit(img)
            center = kmeans.cluster_centers_
            label = kmeans.labels_ 
    
            del kmeans
    
            numLabels = np.arange(0, CLUSTERS + 1)
            (hist, _) = np.histogram(label, bins=numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()
            colors = center[(-hist).argsort()]
        # c = connection.cursor()
        #color_ints = []
    
        logging.debug("{}: {}".format(filen, str(colors)))
        with open("{}.txt".format(filen), 'w') as fd:
            for i in range(CLUSTERS):
                col = int(colors[i][2]) << 16 | int(colors[i][1]) << 8 | int(colors[i][0])
                assert col <= 2**24
                # color_ints.append(col)
                fd.write("{}\n".format(str(col)))
    
        #logging.debug("{}: {}".format(file, str(color_ints)))
    
        #c.execute("INSERT INTO images VALUES (\"{}\",{})".format(os.path.basename(file),",".join([str(i) for i in color_ints])))
        #connection.commit()
        #c.close()
        #images.update({os.path.basename(file): color_ints})
    except Exception as e:
        logging.error(str(e))
        pass

def processTar(tarball):
    if not tarball.endswith(".tar"):
        return

    images = dict()

    import tarfile
    f = tarfile.open(tarball, 'r')
    dirn = "/dev/shm/{}".format(os.path.basename(tarball))
    try:
        os.mkdir(dirn)

        f.extractall(path=dirn)
    except FileExistsError:
        pass

    if DEBUG:
        for img in os.listdir(dirn):
            start=datetime.now()
            process(os.path.join(dirn,img))
            print("{}: Time: {}".format(os.path.basename(img), datetime.now()-start))
    else:
        files = os.listdir(dirn)
        Parallel(n_jobs=150, backend="multiprocessing", batch_size=10, verbose=10)(
            delayed(process)(os.path.join(dirn,img)) for img in files)

    for f in os.listdir(dirn):
        if f.endswith(".txt"):
            with open(os.path.join(dirn, f), 'r') as fd:
                lines = fd.readlines()
                color_ints = [int(n.strip()) for n in lines]
                images.update({f[:-4]: color_ints})
    
    connection = sqlite3.connect("hist_{}_{}.sqlite".format(os.path.basename(tarball),db), check_same_thread = False)
    c = connection.cursor()
    c.execute("CREATE TABLE images (imgname text)")
    connection.commit()
    for i in range(CLUSTERS):
        c.execute("ALTER TABLE images ADD COLUMN color{} integer".format(i))

    connection.commit()
    for key, value in images.items():
        c.execute("INSERT INTO images VALUES (\"{}\",{})".format(key,
                                                                 ",".join([str(i) for i in value])))
    connection.commit()
    connection.close()

    shutil.rmtree(dirn)


def main(folder, img, db, dst):
    times = []
    ctr = 0
    amnt = len(os.listdir(folder))
    for tarball in os.listdir(folder):
        ctr += 1
        print("\n\n\nProcessing {}: {}/{} ({}%) \n\n\n\n".format(tarball, ctr, amnt, (float(ctr)/amnt)*100))
        if os.path.exists(os.path.join(dst,"hist_{}_{}.sqlite".format(tarball,db))):
            print("{} already done. skipping.".format(tarball))
            continue
        start = datetime.now()
        try:
            processTar(os.path.join(folder,tarball))
        except:
            print("{} failed. skipping.".format(tarball))
            shutil.rmtree(os.path.join("/dev/shm", tarball))
        else:
            shutil.copyfile(
                "hist_{}_{}.sqlite".format(tarball,db), 
                os.path.join(dst,"hist_{}_{}.sqlite".format(tarball,db))
            )
            end = datetime.now()
            times.append(end-start)
            print("\n\n{} took {} - AVG {} - TTE {}, ETA {}".format(
                tarball, 
                end-start, 
                sum(times, timedelta(0)) / len(times),
                sum(times, timedelta(0)),
                (sum(times, timedelta(0)) / len(times)) * (amnt - ctr)
            ))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(sys.path)
    assert len(sys.argv) > 4
    folder = sys.argv[1]
    img = sys.argv[2]
    db = sys.argv[3]
    dst = sys.argv[4]
    main(folder, img, db, dst)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

