#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"
import os
from datetime import datetime
from math import ceil
from multiprocessing.pool import Pool
from multiprocessing import Queue

import keras
import numpy as np
import scipy as sp
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from aptos_inference import process_wrapper, preprocess_one_image


# __________________________________________________
# Config, set by user.

MODEL_NAME = 'aptos-training-jing'
DATASET_DIR = '../input/aptos2019-blindness-detection'
IMAGE_SHAPE = (224, 224, 3)

# ##
TEST_CSV_PATH = os.path.join(DATASET_DIR, 'test.csv')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test_images')
MODEL_PATH = f'../input/{MODEL_NAME}/model.h5'

def main():
    # print("loading model...")
    # model = keras.models.load_model(MODEL_PATH)
    # model.summary()
    # __________________________________________________
    # Preprocessing...
    tic = datetime.now()
    # read csv ----> test_df
    test_df = pd.read_csv(TEST_CSV_PATH)
    ids = test_df['id_code']
    # multi-processing
    images_per_process = ceil(len(ids) / os.cpu_count())
    pool = Pool()
    result_queue = Queue()
    for i in range(os.cpu_count()):
        sub_ids = ids[i * images_per_process : (i + 1) * images_per_process]
        pool.apply_async(
            process_wrapper,
            args=(TEST_IMAGES_DIR, sub_ids, preprocess_one_image, result_queue, )
        )
    pool.close()
    pool.join()
    # reassemble image array ----> x_test
    x_test = np.empty((len(ids), *IMAGE_SHAPE), dtype=np.uint8)
    while not result_queue.empty():
        idx, img_array = result_queue.get()
        x_test[idx.min():idx.max(), ...] = img_array
    toc = datetime.now(); print("preprocess_images spend: %f s" % (toc - tic).total_seconds())

    # tic = datetime.now()
    # y_regression = model.predict(x_test)
    # toc = datetime.now(); print("model.predict spend: %f s" % (toc - tic).total_seconds())
    #
    # y_pred = kappa_opt.predict(y_regression)
    #
    # test_df = pd.DataFrame()
    # test_df['id_code'] = ids
    # test_df['diagnosis'] = y_pred
    # test_df.to_csv('submission.csv', index=False)
    # print(test_df)



if __name__ == '__main__':
    main()