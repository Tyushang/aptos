#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"
import abc
import json
from datetime import datetime
from multiprocessing import Pool
from os import path

import keras
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ______________________________________________________________________________
# Config:
# set by user:
MODEL_NAME = 'aptos-training-jing'
DATASET_DIR = '../input/aptos2019-blindness-detection'
# auto-generated or doesn't need to change:
MODEL_PATH = f'../input/{MODEL_NAME}/model.h5'
PREP_ALGO_SRC_PATH = '../input/aptos-preprocessed/prep_algo.py'
INP_CONFIG_PATH = f'../input/{MODEL_NAME}/config.json'
with open(INP_CONFIG_PATH) as f:
    config = json.load(f)
IMAGE_SHAPE = config['image_shape']
KAPPA_COEF = config['kappa_coef']

# ______________________________________________________________________________
# preprocessing_algorithm for one image:
# def preprocessing_algorithm(inp_img, desired_size=224, resample=Image.LANCZOS):
#     return inp_img.resize((desired_size,) * 2, resample=resample)

# ______________________________________________________________________________
# Preprocessing Class:


class PreprocessBase():

    def __init__(self, img_ids, img_per_chunk=100):
        self.img_ids = img_ids
        self.chunk_size = img_per_chunk

    def do_prep(self):
        with Pool() as pool:
            res = list(tqdm(pool.imap(self.prep_one_img, self.img_ids, self.chunk_size)))
        return res

    @abc.abstractmethod
    def prep_one_img(self, img_id): ...

    def before_prep(self, **kw): ...

    def post_prep(self, **kw): ...


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

# ______________________________________________________________________________
# Functions:


def classify(regression_preds, coef=(0.5, 1.5, 2.5, 3.5)):
    """
    using numpy broadcasting feature to simplify code.
    see: https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#ufuncs-broadcasting
    :param regression_preds:
    :param coef:
    :return:
    """
    return (regression_preds.reshape([-1, 1]) > np.array(coef).reshape([1, -1])).sum(axis=1)


# In[]: main process:
# ______________________________________________________________________________
# loading model...
print("loading model...")
model = keras.models.load_model(MODEL_PATH)
model.summary()

# ______________________________________________________________________________
# preprocess
with open(PREP_ALGO_SRC_PATH, 'r') as f:
    exec(f.read())
    handler = eval('preprocessing_algorithm')

prep = Prep4Test(
    dataset_dir=DATASET_DIR,
    prep_algo_per_img=handler,
)

print("==== now preprocessing...")
tic = datetime.now()
x_test = np.array(prep.do_prep())
toc = datetime.now(); print("do_prep spend: %f s" % (toc - tic).total_seconds())
test_df = prep.test_df

# ______________________________________________________________________________
# predict and submit
tic = datetime.now()
y_regression = model.predict(x_test)
toc = datetime.now(); print("model.predict spend: %f s" % (toc - tic).total_seconds())

y_pred = classify(y_regression, KAPPA_COEF)

test_df['diagnosis'] = y_pred
test_df.to_csv('submission.csv', index=False)
print(test_df)


