#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import abc
import inspect
import json
import os
import shutil
import tarfile
from datetime import datetime
from multiprocessing import Pool
from os import path

import pandas as pd
import pytz
from tqdm import tqdm

# ______________________________________________________________________________
# Config:
# set by user:
DATASET_DIR = '../input'
IMAGE_SHAPE = [224, 224, 3]
OUT_TAR_FILE = './aptos-preprocessed.tar'
ARCNAME = 'aptos-preprocessed'
# auto-generated or doesn't need to change:
CACHE_DIR = '/tmp/aptos-preprocessed'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

config = {
    'image_shape': IMAGE_SHAPE
}

# ______________________________________________________________________________
# Preprocessing Class:


class PreprocessBase():

    def __init__(self, img_ids, img_per_chunk=100):
        self.img_ids = img_ids
        self.chunk_size = img_per_chunk

    def do_prep(self):
        with Pool() as pool:
            res = list(tqdm(
                pool.imap(self.prep_one_img, self.img_ids, self.chunk_size)
            ))
        return res

    @abc.abstractmethod
    def prep_one_img(self, img_id): ...

    def before_prep(self, **kw): ...

    def post_prep(self, **kw): ...


class Prep4Train(PreprocessBase):

    def __init__(self, dataset_dir, prep_algo_per_img, img_per_chunk=100,
                 cache_dir=CACHE_DIR, out_tar_file=OUT_TAR_FILE, arcname=ARCNAME):
        img_ids = pd.read_csv(path.join(dataset_dir, 'train.csv'))['id_code']
        super().__init__(img_ids, img_per_chunk)
        #
        self.dataset_dir = dataset_dir
        self.img_dir = path.join(dataset_dir, 'train_images')
        self.prep_algo_per_img = prep_algo_per_img
        self.cache_dir = cache_dir
        self.out_tar_file = out_tar_file
        self.arcname = arcname

    def prep_one_img(self, img_id):
        img_path = path.join(self.img_dir, img_id + '.png')
        out_path = path.join(self.cache_dir, 'train_images', img_id + '.png')
        # apply preprocessing algo, convert color & save.
        from cv2 import cv2
        img = self.prep_algo_per_img(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img)

    def before_prep(self, **kw):
        if not path.exists(path.join(self.cache_dir, 'train_images')):
            os.makedirs(path.join(self.cache_dir, 'train_images'), exist_ok=True)

    def post_prep(self, **kw):
        # copy .csv file to cache dir for later use.
        for csv_name in [x for x in os.listdir(self.dataset_dir) if path.splitext(x)[1] == '.csv']:
            shutil.copyfile(path.join(self.dataset_dir, csv_name),
                            path.join(self.cache_dir, csv_name))
        # # serialize preprocess algorithm for each image to cache dir for later use.
        # with open(path.join(self.cache_dir, 'prep_algo.pickle'), 'wb') as f:
        #     cloudpickle.dump(self.prep_algo_per_img, f)
        # save source code of self.prep_algo_per_img for later use.
        with open(path.join(self.cache_dir, 'prep_algo.py'), 'w') as f:
            src = inspect.getsource(self.prep_algo_per_img)
            f.write(src)
        # dump config to cache dir for later use.
        with open(path.join(self.cache_dir, 'config.json'), 'w') as f:
            config['create_time'] = str(datetime.now(tz=REMOTE_TZ))
            json.dump(config, f)
        # tar cache dir to OUT_TAR_FILE for later use.
        with tarfile.open(self.out_tar_file, 'w') as tar:
            tar.add(self.cache_dir, arcname=self.arcname)
            print("================ content of tar file ================")
            tar.list()


class Prep4Test(PreprocessBase):

    def __init__(self, dataset_dir, prep_algo_per_img, img_per_chunk=100):
        self.test_df = pd.read_csv(path.join(dataset_dir, 'test.csv'))
        img_ids = self.test_df['id_code']
        super().__init__(img_ids, img_per_chunk)
        self.img_dir = path.join(dataset_dir, 'test_images')
        self.prep_algo_per_img = prep_algo_per_img

    def prep_one_img(self, img_id):
        img_path = path.join(self.img_dir, img_id + '.png')
        return self.prep_algo_per_img(img_path)


# In[]:
# ______________________________________________________________________________
# main process:

# ______________________________________________________________________________
# #### for train:
# def preprocessing_algorithm(img_path, desired_size=224, resample=Image.LANCZOS):
#     img = Image.open(img_path)
#     return np.array(img.resize((desired_size,) * 2, resample=resample))

def preprocessing_algorithm(img_path, sigmaX=10):
    from cv2 import cv2
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = crop_image_from_gray(image)
    image = cv2.resize(image, tuple(IMAGE_SHAPE[:-1]))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


prep = Prep4Train(
    dataset_dir=DATASET_DIR,
    prep_algo_per_img=preprocessing_algorithm,
    cache_dir=CACHE_DIR,
    out_tar_file=OUT_TAR_FILE
)

prep.before_prep()
print("==== now preprocessing...")
tic = datetime.now()
prep.do_prep()
toc = datetime.now(); print("do_prep spend: %f s" % (toc - tic).total_seconds())
prep.post_prep()

# In[]:
# ______________________________________________________________________________
# #### for test:


# with open(path.join(CACHE_DIR, 'prep_algo.py'), 'r') as f:
#     src = f.read()
#     exec(src)
#     handler = eval('preprocessing_algorithm')
#
# prep = Prep4Test(
#     dataset_dir=DATASET_DIR,
#     prep_algo_per_img=handler,
# )
#
# print("==== now preprocessing...")
# tic = datetime.now()
# x_test = np.array(prep.do_prep())
# toc = datetime.now(); print("do_prep spend: %f s" % (toc - tic).total_seconds())







