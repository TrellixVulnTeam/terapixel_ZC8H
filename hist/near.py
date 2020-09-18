#!/usr/bin/env python3

import sys, os
import sqlite3
import numpy as np
import cv2
import pandas as pd
import cudf
# from sklearn.neighbors import NearestNeighbors
from pai4sk.neighbors import NearestNeighbors
from collections import defaultdict
import random
import functools
import json
import h5py
from datetime import datetime
import copy

print = functools.partial(print, flush=True, file=sys.stderr)

# dirn = os.path.dirname(os.path.realpath(__file__))
dirn = "/data/others/bbecker"

img_to_process = sys.argv[1]

print(img_to_process)

array = []
col_dict = defaultdict(list)


def arrpack(arr):
    return (arr[0] & 255) << 16 | (arr[1] & 255) << 8 | (arr[2] & 255)


def arrunpack(num):
    return [(num >> 16) & 255, (num >> 8) & 255, (num & 255)]


def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d' % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf


for f in os.listdir(os.path.join(dirn, "tp_data")):
    if f.endswith(".sqlite"):
        # print("{}\n".format(os.path.join(dirn,"tp_data",f)))
        print(".", end='', flush=True)
        try:
            connection = sqlite3.connect(os.path.join(dirn, "tp_data", f))
            c = connection.cursor()
            for row in c.execute("SELECT color1, imgname FROM images"):
                # r = row[0] >> 16 & 255
                # g = row[0] >> 8  & 255
                # b = row[0]       & 255
                # array.append([r, g, b])
                array.append(row[0])
                col_dict[row[0]].append(row[1])
            connection.close()
        except sqlite3.OperationalError:
            pass

print("")
if os.path.exists("{}-dump.hdf5".format(img_to_process)):
    print("dump.hdf5 exists, reading data...")
    with h5py.File("{}-dump.hdf5".format(img_to_process), 'r') as f:
        print(list(f.keys()))
        array = np.array(f["array"])
        array_p = np.array(f["array_p"])
        img = np.array(f["img"])
        img_p = np.array(f["img_p"])
        img_p_u = np.array(f["img_p_u"])
        img_p_u_inverse = np.array(f["img_p_u_inverse"])
        img_p_u_counts = np.array(f["img_p_u_counts"])
        array_p_u = np.array(f["array_p_u"])
        array_p_u_inverse = np.array(f["array_p_u_inverse"])
        array_p_u_counts = np.array(f["array_p_u_counts"])
        img_p_u_um = np.array(f["img_p_u_um"])
        img_p_u_um_counts = np.array(f["img_p_u_um_counts"])
        array_p_u_um = np.array(f["array_p_u_um"])
        array_p_u_um_counts = np.array(f["array_p_u_um_counts"])
        data_p_u_m = np.array(f["data_p_u_m"])
        img_p_u_mi = np.array(f["img_p_u_mi"])
        array_p_u_mi = np.array(f["array_p_u_mi"])
        img_rest = np.array(f["img_rest"])
        img_rest_counts = np.array(f["img_rest_counts"])
        array_rest = np.array(f["array_rest"])
        array_rest_counts = np.array(f["array_rest_counts"])
        search_space_ = np.array(f["search_space_"])
        search_space = np.array(f["search_space"])
        find_points_ = np.array(f["find_points_"])
        find_points = np.array(f["find_points"])
        distances = np.array(f["distances"])
        indices = np.array(f["indices"])
        old_shape = cv2.imread(img_to_process).shape
else:

    # print(json.dumps(
    #    col_dict,
    #    sort_keys=True,
    #    indent=4,
    #    separators=(',', ': ')
    # ))

    # sys.exit(0)

    array = np.array(array)

    print("cv2.imread()")
    img = cv2.cvtColor(cv2.imread(img_to_process), cv2.COLOR_BGR2RGB)

    old_shape = img.shape
    print("img.reshape()")
    img = img.reshape((old_shape[0] * old_shape[1], old_shape[2]))
    # print(img)

    print("arrpack()")
    # array_p = np.array([arrpack(p) for p in array])
    array_p = array
    img_p = np.array([arrpack(p) for p in img])

    print("np.unique(img)")
    img_p_u, img_p_u_inverse, img_p_u_counts = np.unique(img_p, return_counts=True, return_inverse=True, axis=0)
    print("np.unique(array)")
    array_p_u, array_p_u_inverse, array_p_u_counts = np.unique(array_p, return_counts=True, return_inverse=True, axis=0)

    # for t in zip(array_p_u, array_p_u_counts):
    #    print(t)

    # sys.exit(0)

    print("np.setdiff1d(img_p_u, array_p_u)")
    img_p_u_um = np.setdiff1d(img_p_u, array_p_u, assume_unique=True)
    img_p_u_um_counts = img_p_u_counts[np.in1d(img_p_u, img_p_u_um)]

    print("np.setdiff1d(array_p_u, img_p_u)")
    array_p_u_um = np.setdiff1d(array_p_u, img_p_u, assume_unique=True)
    array_p_u_um_counts = array_p_u_counts[np.in1d(array_p_u, array_p_u_um)]

    print("np.intersect1d(img_p_u, array_p_u)")
    data_p_u_m, img_p_u_mi, array_p_u_mi = np.intersect1d(img_p_u, array_p_u, assume_unique=True, return_indices=True)

    img_rest = []
    img_rest_counts = []
    array_rest = []
    array_rest_counts = []

    print("calulation of rests")
    for (dp, i, j) in zip(data_p_u_m, img_p_u_mi, array_p_u_mi):
        if img_p_u_counts[i] < array_p_u_counts[j]:
            array_rest.append(dp)
            array_rest_counts.append(array_p_u_counts[j] - img_p_u_counts[i])
        elif img_p_u_counts[i] > array_p_u_counts[j]:
            img_rest.append(dp)
            img_rest_counts.append(img_p_u_counts[i] - array_p_u_counts[j])

    img_rest = np.array(img_rest)
    img_rest_counts = np.array(img_rest_counts)
    array_rest = np.array(array_rest)
    array_rest_counts = np.array(array_rest_counts)

    print("{} pixels, {} candidates".format(img.shape[0], array.shape[0]))
    print("{} colors required, {} colors available".format(img_p_u.shape[0], array_p_u.shape[0]))
    print("{} colors matched, {} matched low, {} matched high".format(data_p_u_m.shape[0], img_rest.shape[0],
                                                                      array_rest.shape[0]))
    print("{} colors unmatched in img, {} colors unmatched in array".format(img_p_u_um.shape[0], array_p_u_um.shape[0]))

    search_space_ = np.union1d(array_p_u_um, array_rest)
    search_space = np.array([arrunpack(p) for p in search_space_])

    find_points_ = np.union1d(img_p_u_um, img_rest)
    find_points = np.array([arrunpack(p) for p in find_points_])

    ntf = int(max(img_rest_counts.max() / np.average(array_rest_counts),
                  img_p_u_um_counts.max() / np.average(array_p_u_um_counts)))

    print("fit(search_space) clr/{}".format(ntf))
    nn = NearestNeighbors(n_neighbors=ntf, algorithm='auto', n_jobs=155).fit(search_space)
    print(nn)
    print("kneighbors(find_points)")
    distances, indices = nn.kneighbors(find_points)

    print(distances)
    print(indices)

    print("saving data to {}-dump.hdf5".format(img_to_process))
    with h5py.File("{}-dump.hdf5".format(img_to_process), "w") as f:
        f.create_dataset("array", data=array)
        f.create_dataset("array_p", data=array_p)
        f.create_dataset("img", data=img)
        f.create_dataset("img_p", data=img_p)
        f.create_dataset("img_p_u", data=img_p_u)
        f.create_dataset("img_p_u_inverse", data=img_p_u_inverse)
        f.create_dataset("img_p_u_counts", data=img_p_u_counts)
        f.create_dataset("array_p_u", data=array_p_u)
        f.create_dataset("array_p_u_inverse", data=array_p_u_inverse)
        f.create_dataset("array_p_u_counts", data=array_p_u_counts)
        f.create_dataset("img_p_u_um", data=img_p_u_um)
        f.create_dataset("img_p_u_um_counts", data=img_p_u_um_counts)
        f.create_dataset("array_p_u_um", data=array_p_u_um)
        f.create_dataset("array_p_u_um_counts", data=array_p_u_um_counts)
        f.create_dataset("data_p_u_m", data=data_p_u_m)
        f.create_dataset("img_p_u_mi", data=img_p_u_mi)
        f.create_dataset("array_p_u_mi", data=array_p_u_mi)
        f.create_dataset("img_rest", data=img_rest)
        f.create_dataset("img_rest_counts", data=img_rest_counts)
        f.create_dataset("array_rest", data=array_rest)
        f.create_dataset("array_rest_counts", data=array_rest_counts)
        f.create_dataset("search_space_", data=search_space_)
        f.create_dataset("search_space", data=search_space)
        f.create_dataset("find_points_", data=find_points_)
        f.create_dataset("find_points", data=find_points)
        f.create_dataset("distances", data=distances)
        f.create_dataset("indices", data=indices)

new_img = []
new_img_fns = []

fns_blacklist = []

col_times_used = defaultdict(int)
alt_col_times_used = defaultdict(int)
# array_p_u_counts

duplicats_used = False

col_dict_work = copy.deepcopy(col_dict)

print("start matching phase")
try:
    for (pix, ctr) in zip(img_p_u_inverse, range(img_p_u_inverse.shape[0])):
        start = datetime.now()
        # pix = index of color in img_p_u
        # case 1: there is a sufficiant number of perfect matches
        # case 2: there are perfect matches, but not enough
        # case 3: there are only approximations available
        # case 2a and 3a: the number of approximations is to low -> we are fucked!
        print("Pixel {} ".format(ctr), end='', flush=True)
        curr_col = img_p_u[pix]
        print("Color {} ".format(curr_col), end='', flush=True)
        curr_col_in_intersect = (data_p_u_m == curr_col).nonzero()[0]
        if curr_col_in_intersect.shape[0] > 0:
            print("matched ", end='', flush=True)
            # our color is in the intersection of img_p_u and array_p_u
            ind_in_intersect = curr_col_in_intersect[0]
            colamount_in_img = img_p_u_counts[pix]
            colamount_in_arr = array_p_u_counts[array_p_u_mi[ind_in_intersect]]
            if colamount_in_img <= colamount_in_arr:
                # we are in case 1
                col_times_used[curr_col] += 1
                print("perfectly ", end='', flush=True)
                new_img.append(arrunpack(curr_col))
                # choose_from = list(set(col_dict[curr_col]) - set(fns_blacklist))
                choose_from = col_dict_work[curr_col]
                print("{} avail for color, {} used for color, {} left for color, {} total blacklisted ".format(
                    len(col_dict[curr_col]), col_times_used[curr_col], len(choose_from), len(fns_blacklist)), end='',
                      flush=True)
                fn = choose_from.pop(random.randrange(len(choose_from)))
                col_dict_work[curr_col] = choose_from
                print("file {} ".format(fn), end='', flush=True)
                fns_blacklist.append(fn)
                new_img_fns.append(fn)
            else:
                # we are in case 2
                num_of_perf_match = colamount_in_arr
                print("low ", end='', flush=True)
                # can we still use a perfect one?
                if col_times_used[curr_col] < num_of_perf_match:
                    # flip a slightly biased (2:3) coin to decide whether to use one
                    print("perfects available ", end='', flush=True)
                    if random.randint(0, 2) % 2 == 0:
                        print("using perfect ", end='', flush=True)
                        col_times_used[curr_col] += 1
                        new_img.append(arrunpack(curr_col))
                        # choose_from = list(set(col_dict[curr_col]) - set(fns_blacklist))
                        choose_from = col_dict_work[curr_col]
                        print("{} used for color, {} left for color, {} total blacklisted ".format(
                            col_times_used[curr_col], len(choose_from), len(fns_blacklist)), end='', flush=True)
                        fn = choose_from.pop(random.randrange(len(choose_from)))
                        col_dict_work[curr_col] = choose_from
                        print("file {} ".format(fn), end='', flush=True)
                        fns_blacklist.append(fn)
                        new_img_fns.append(fn)
                    else:
                        # use an alternative
                        print("using alt ", end='', flush=True)
                        ind_in_find_points = (find_points_ == curr_col).nonzero()[0][0]
                        alt_indices = indices[ind_in_find_points]  # list of indexes in search space
                        alt_colors = search_space_[np.sort(alt_indices)]
                        col_counts = array_p_u_counts[np.isin(array_p_u, alt_colors)]
                        col_assigned = False
                        for (alt_col, alt_col_count) in zip(alt_colors, col_counts):
                            is_reserved = False
                            if (img_p_u == alt_col).nonzero()[0].shape[0] > 0:
                                # alternate color candiate is also in img
                                h = img_p_u_counts[(img_p_u == alt_col).nonzero()[0][0]]
                                if h < alt_col_count - alt_col_times_used[alt_col]:
                                    is_reserved = False
                                else:
                                    # we still need this color for remaining perfect matches
                                    is_reserved = True
                            if alt_col_times_used[alt_col] < alt_col_count and not is_reserved:
                                print("{} ".format(alt_col), end='', flush=True)
                                alt_col_times_used[alt_col] += 1
                                new_img.append(arrunpack(alt_col))
                                # choose_from = list(set(col_dict[alt_col]) - set(fns_blacklist))
                                choose_from = col_dict_work[alt_col]
                                print("{} avail for alt, {} used of alt, {} left for alt, {} total blacklisted ".format(
                                    alt_col_count, alt_col_times_used[alt_col], len(choose_from), len(fns_blacklist)),
                                      end='', flush=True)
                                fn = choose_from.pop(random.randrange(len(choose_from)))
                                col_dict_work[alt_col] = choose_from
                                print("file {} ".format(fn), end='', flush=True)
                                fns_blacklist.append(fn)
                                new_img_fns.append(fn)
                                col_assigned = True
                                break
                        if not col_assigned:
                            print("failed ", end='', flush=True)
                            # we ran out of alternatives (case 2a)
                            if col_times_used[curr_col] < num_of_perf_match:
                                # but we have perfect matches left. TODO duplicate code
                                print("using perfect ", end='', flush=True)
                                col_times_used[curr_col] += 1
                                new_img.append(arrunpack(curr_col))
                                # choose_from = list(set(col_dict[curr_col]) - set(fns_blacklist))
                                choose_from = col_dict_work[curr_col]
                                print("{} used for color, {} left for color, {} total blacklisted ".format(
                                    col_times_used[curr_col], len(choose_from), len(fns_blacklist)), end='', flush=True)
                                fn = choose_from.pop(random.randrange(len(choose_from)))
                                col_dict_work[curr_col] = choose_from
                                print("file {} ".format(fn), end='', flush=True)
                                fns_blacklist.append(fn)
                                new_img_fns.append(fn)
                            else:
                                # we need to take a duplicate :(
                                print("using duplicate perfect ", end='', flush=True)
                                duplicate_used = True
                                new_img.append(arrunpack(curr_col))
                                fn = random.choice(col_dict[curr_col])
                                print("file {} ".format(fn), end='', flush=True)
                                new_img_fns.append(fn)
                else:
                    # no unused perfects available anymore. TODO duplicate code :(
                    # use an alternative
                    print("using alt ", end='', flush=True)
                    ind_in_find_points = (find_points_ == curr_col).nonzero()[0][0]
                    alt_indices = indices[ind_in_find_points]  # list of indexes in search space
                    alt_colors = search_space_[np.sort(alt_indices)]
                    col_counts = array_p_u_counts[np.isin(array_p_u, alt_colors)]
                    col_assigned = False
                    for (alt_col, alt_col_count) in zip(alt_colors, col_counts):
                        if (img_p_u == alt_col).nonzero()[0].shape[0] > 0:
                            is_reserved = False
                            # alternate color candiate is also in img
                            h = img_p_u_counts[(img_p_u == alt_col).nonzero()[0][0]]
                            if h < alt_col_count - alt_col_times_used[alt_col]:
                                is_reserved = False
                            else:
                                # we still need this color for remaining perfect matches
                                is_reserved = True
                        if alt_col_times_used[alt_col] < alt_col_count and not is_reserved:
                            print("{} ".format(alt_col), end='', flush=True)
                            alt_col_times_used[alt_col] += 1
                            new_img.append(arrunpack(alt_col))
                            # choose_from = list(set(col_dict[alt_col]) - set(fns_blacklist))
                            choose_from = col_dict_work[alt_col]
                            print("{} used of alt, {} left for alt, {} total blacklisted ".format(
                                alt_col_times_used[alt_col], len(choose_from), len(fns_blacklist)), end='', flush=True)
                            fn = choose_from.pop(random.randrange(len(choose_from)))
                            col_dict_work[alt_col] = choose_from
                            print("file {} ".format(fn), end='', flush=True)
                            fns_blacklist.append(fn)
                            new_img_fns.append(fn)
                            col_assigned = True
                            break
                    if not col_assigned:
                        print("failed ", end='', flush=True)
                        # we need to take a duplicate :(
                        print("using duplicate perfect ", end='', flush=True)
                        duplicate_used = True
                        new_img.append(arrunpack(curr_col))
                        fn = random.choice(col_dict[curr_col])
                        print("file {} ".format(fn), end='', flush=True)
                        new_img_fns.append(fn)

        else:
            # our color is in the intersection of img_p_u and array_p_u, therefore we are in case 3
            # TODO kinda dup of 2
            print("unmatched ", end='', flush=True)
            ind_in_find_points = (find_points_ == curr_col).nonzero()[0][0]
            alt_indices = indices[ind_in_find_points]  # list of indexes in search space
            alt_colors = search_space_[np.sort(alt_indices)]
            col_counts = array_p_u_counts[np.isin(array_p_u, alt_colors)]
            col_assigned = False
            for (alt_col, alt_col_count) in zip(alt_colors, col_counts):
                is_reserved = False
                if (img_p_u == alt_col).nonzero()[0].shape[0] > 0:
                    # alternate color candiate is also in img
                    h = img_p_u_counts[(img_p_u == alt_col).nonzero()[0][0]]
                    if h < alt_col_count - alt_col_times_used[alt_col]:
                        is_reserved = False
                    else:
                        # we still need this color for remaining perfect matches
                        is_reserved = True
                if alt_col_times_used[alt_col] < alt_col_count and not is_reserved:
                    alt_col_times_used[alt_col] += 1
                    new_img.append(arrunpack(alt_col))
                    # choose_from = list(set(col_dict[alt_col]) - set(fns_blacklist))
                    choose_from = col_dict_work[alt_col]
                    print("{} used of alt, {} left for alt, {} total blacklisted ".format(alt_col_times_used[alt_col],
                                                                                          len(choose_from),
                                                                                          len(fns_blacklist)), end='',
                          flush=True)
                    fn = choose_from.pop(random.randrange(len(choose_from)))
                    col_dict_work[alt_col] = choose_from
                    print("file {} ".format(fn), end='', flush=True)
                    fns_blacklist.append(fn)
                    new_img_fns.append(fn)
                    col_assigned = True
                    break
            if not col_assigned:
                # we ran out of alternatives (case 3a)
                # we need to take a duplicate :(
                print("duplicate required, ", end='', flush=True)
                duplicate_used = True
                col_to_use = random.choice(alt_colors)
                new_img.append(arrunpack(col_to_use))
                fn = random.choice(col_dict[col_to_use])
                print("file {} ".format(fn), end='', flush=True)
                new_img_fns.append(fn)
        print("")
        print(datetime.now() - start)
except:
    for _ in range(img.shape[0] - len(new_img)):
        new_img.append([0, 0, 0])
    new_img = np.array(new_img).astype("uint8")
    new_img = new_img.reshape(old_shape)
    new_img = cv2.cvtColor(new_img.astype("uint8"), cv2.COLOR_BGR2RGB)
    cv2.imwrite("{}-crash.jpg".format(img_to_process), new_img)
    print("delete {}-dump.hdf5 to start over".format(img_to_process))
    raise

print("\n\n")

if duplicate_used:
    print("BAD NEWS: duplicates needed to be used :(")

new_img = np.array(new_img)
new_img = new_img.reshape(old_shape)

new_img_fns = np.array(new_img_fns)
new_img_fns = new_img_fns.reshape((old_shape[0], old_shape[1]))

print(new_img_fns)
try:
    new_img = new_img.astype("uint8")
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("{}-result.jpg".format(img_to_process), new_img)
except:
    pass

try:
    np.savetxt("{}-dump.csv".format(img_to_process), new_img_fns, delimiter=',')
except:
    pass

try:
    np.save("{}-dump.npy".format(img_to_process), new_img_fns)
except:
    pass

try:
    new_img_fns.tofile("{}-dump.dat".format(img_to_process), sep=',')
except:
    pass

with h5py.File("{}-dump.hdf5".format(img_to_process), 'a') as f:
    try:
        f.create_dataset("new_img", data=new_img)
    except:
        pass
    new_img_fns_h5 = np.char.decode(new_img_fns.astype(np.bytes_), 'UTF-8')
    f.create_dataset("new_img_fns", data=new_img_fns_h5)

sys.exit(0)
