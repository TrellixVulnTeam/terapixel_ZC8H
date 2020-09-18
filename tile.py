#!/usr/bin/env python3

from PIL import Image
#import h5py
import sys, os
import numpy as np
from math import ceil
import tarfile
import random
import b2sdk.v1 as b2
from datetime import datetime


#with h5py.File(sys.argv[1], 'r') as f:
#    new_img_fns = np.array(f["new_img_fns"])

new_img_fns = np.load(sys.argv[1], allow_pickle=True)


name_base = "terrapixel/{}".format(sys.argv[1])
frames_data = "/data/others/bbecker/frames"


info = b2.InMemoryAccountInfo()
b2_api = b2.B2Api(info)
application_key_id = '0000000000000000000000000' #TODO
application_key = 'rathecai7ooJ0oogh2eet1oitoo2deiY' #TODO
b2_api.authorize_account("production", application_key_id, application_key)
b2_bucket = b2_api.get_bucket_by_name("REDACTED")

for h in range(8):
    ratio = pow(4, h) # zoom is 1:ratio
    cols_rows = pow(2, h) # cols_rows x cols_rows on one canvas
    level = 8 - h
    print("LEVEL RATIO 1:{}".format(ratio))
    startlevel = datetime.now()
    for i in range(int(ceil(new_img_fns.shape[0] / cols_rows))):
        startrow = datetime.now()
        for j in range(int(ceil(new_img_fns.shape[1] / cols_rows))):
            startcol = datetime.now()
            # canvas cell
            loc_n = "{}/{}/{}/{}.jpg".format(name_base, level, i, j)
            img_width = int(ceil(1920/cols_rows))
            img_height = int(ceil(1080/cols_rows))
            base_canvas = Image.new("RGB", (1920, 1080))
            for k in range(cols_rows):
                for l in range(cols_rows):
                    img_fn = new_img_fns[i * cols_rows + k, j * cols_rows + l]
                    print(img_fn)
                    img_path = os.path.join(frames_data, img_fn[:-len("_000000.jpg")])
                    if not os.path.exists(img_path):
                        print("extracting tarball")
                        os.mkdir(img_path)
                        tarball = os.path.join(frames_data, "{}.tar".format(img_fn[:-len("_000000.jpg")]))
                        with tarfile.open(tarball, 'r') as tar:
                            tar.extractall(path=img_path)
                        os.unlink(tarball)
                    img = Image.open(os.path.join(img_path, img_fn))
                    if img.size[0]/float(img.size[1]) > 16.0/9.0:
                        new_width = int(ceil(img.size[1] * 16/float(9)))
                        offset = int(ceil((img.size[0] - new_width)/2.0))
                        img = img.crop((offset, 0, new_width, img.size[1]))
                    # resize to level size
                    img = img.resize((img_width, img_height), resample=Image.BICUBIC)
                    base_canvas.paste(img, (img_width*l, img_height*k))
            pre = random.randint(10000, 99999)
            tn = os.path.join("/dev/shm", "{}{}".format(pre, loc_n.replace("/", "")))
            base_canvas.save(tn)

            print(b2_bucket.upload_local_file(local_file=tn, file_name=loc_n))

            os.unlink(tn)
            print("COLUMN: {}".format(datetime.now()-startcol))
        print("ROW: {}".format(datetime.now()-startrow))
    print("LEVEL: {}".format(datetime.now()-startlevel))
