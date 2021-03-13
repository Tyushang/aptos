#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import inspect
import json
import os
import tarfile
from datetime import datetime
from math import ceil
from os import path

import numpy as np
import pandas as pd
import pytz
from PIL import Image
from cv2 import cv2
from matplotlib import pyplot as plt

# ______________________________________________________________________________
# Config:
# set by user:
from aptos_utils import PreprocessBase

DATASET_2015_DIR = '../input/diabetic-retinopathy-resized'
DATASET_2019_DIR = '../input/aptos2019-blindness-detection'
CROPPED = False
IMAGE_SHAPE = [224, 224, 3]
OUT_DIR = './'
ARCNAME_2015 = 'aptos-2015-preprocessed'
ARCNAME_2019 = 'aptos-2019-preprocessed'
# auto-generated or doesn't need to change:
if CROPPED:
    IMAGE_2015_DIR = path.join(DATASET_2015_DIR, 'resized_train_cropped', 'resized_train_cropped')
    CSV_2015_PATH = path.join(DATASET_2015_DIR, 'trainLabels_cropped.csv')
else:
    IMAGE_2015_DIR = path.join(DATASET_2015_DIR, 'resized_train', 'resized_train')
    CSV_2015_PATH = path.join(DATASET_2015_DIR, 'trainLabels.csv')
IMAGE_2019_DIR = path.join(DATASET_2019_DIR, 'train_images')
CSV_2019_PATH = path.join(DATASET_2019_DIR, 'train.csv')
CACHE_2019_DIR = '/need-more-space/aptos-2019-preprocessed'
CACHE_2015_DIR = '/need-more-space/aptos-2015-preprocessed'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

config = {}

# ______________________________________________________________________________
#
train_df_2015 = pd.read_csv(CSV_2015_PATH)\
    .rename(columns={'image': 'id_code', 'level': 'diagnosis'})

# ______________________________________________________________________________
# Preprocessing Class:


class Prep4Train(PreprocessBase):

    def __init__(self, train_df, img_dir, img_suffix, prep_algo_per_img, prep_algo_kw=None,
                 img_per_chunk=0x100, out_dir=OUT_DIR, arcname=None, cache_dir=None):
        self.train_df = train_df
        super().__init__(self.train_df['id_code'], img_per_chunk)
        #
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.prep_algo_per_img = prep_algo_per_img
        self.prep_algo_kw = prep_algo_kw if prep_algo_kw is not None else {}
        self.out_dir = out_dir
        self.arcname = arcname
        self.cache_dir = cache_dir

    def prep_one_img(self, img_id):
        img_path = path.join(self.img_dir, img_id + self.img_suffix)
        out_path = path.join(self.cache_dir, 'train_images', img_id + '.png')
        # apply preprocessing algo, convert color & save.
        from cv2 import cv2
        img = self.prep_algo_per_img(img_path, **self.prep_algo_kw)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img)

    def before_prep(self):
        if not path.exists(path.join(self.cache_dir, 'train_images')):
            os.makedirs(path.join(self.cache_dir, 'train_images'), exist_ok=True)

    def post_prep(self):
        # copy train.csv
        self.train_df.to_csv(path.join(self.cache_dir, 'train.csv'), index=False)
        # dump config for later use.
        with open(path.join(self.out_dir, 'config.json'), 'w') as f:
            config['preprocess'] = {
                'image_shape': IMAGE_SHAPE,
                'prep_algo': {
                    'func_name': self.prep_algo_per_img.__name__,
                    'kw': self.prep_algo_kw,
                    'src': inspect.getsource(self.prep_algo_per_img)
                },
                'create_time': str(datetime.now(tz=REMOTE_TZ))
            }
            json.dump(config, f)
        # tar cache dir to OUT_TAR_FILE for later use.
        with tarfile.open(path.join(self.out_dir, self.arcname + '.tar'), 'w') as tar:
            tar.add(self.cache_dir, arcname=self.arcname)

    def show_samples_with_comparison(self, num_sample=20):
        columns = 4
        rows = ceil(num_sample / 2)
        fig = plt.figure(figsize=(5 * columns, 4 * rows))
        for i, t in enumerate(self.train_df.sample(num_sample).itertuples()):
            # image before preprocess
            img_pre = Image.open(path.join(self.img_dir, t.id_code + self.img_suffix))
            fig.add_subplot(rows, columns, 2*i + 1)
            plt.title(t.diagnosis)
            plt.imshow(img_pre)
            # image after preprocess
            img_post = Image.open(path.join(self.cache_dir, 'train_images', t.id_code + '.png'))
            fig.add_subplot(rows, columns, 2*i + 2)
            plt.title(t.diagnosis)
            plt.imshow(img_post)

        plt.tight_layout()
        plt.show()
        plt.close(fig)


# In[]:
# ______________________________________________________________________________
# main process:


# def preprocessing_algorithm(img_path, desired_size=224, resample=Image.LANCZOS):
#     img = Image.open(img_path)
#     return np.array(img.resize((desired_size,) * 2, resample=resample))


def preprocessing_algorithm(img_path, sigmaX=10, interpolation=None):
    """
    borrow from: https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping/comments
    :param interpolation:
    :param img_path:
    :param sigmaX:
    :return:
    """
    def _crop_image_from_gray(img, tol=7):
        import numpy as np
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:  # image is too dark so that we crop out everything,
                return img  # return original image
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                #         print(img1.shape,img2.shape,img3.shape)
                img = np.stack([img1, img2, img3], axis=-1)
            #         print(img.shape)
            return img

    def _standardize(image_array):
        arr = np.asarray(image_array, dtype='float64')
        mean, std = arr.mean(), arr.std()
        return ((arr - mean) / std).astype('float32')

    def _clahe_rgb(rgb_array, clip_limit=2.0, tile_grid_size=(8, 8)):
        # convert RGB to LAB
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        # apply clahe on LAB's L component.
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        # remap LAB tp RGB.
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _crop_image_from_gray(image)
    cv2.normalize(src=image, dst=image, alpha=255, norm_type=cv2.NORM_INF)
    image = cv2.resize(image, tuple(IMAGE_SHAPE[:-1]), interpolation)
    # image = _clahe_rgb(image, clip_limit=5, tile_grid_size=(4,4))
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)# , cv2.INTER_LANCZOS4

    return image


# ______________________________________________________________________________
# #### for 2015:
prep2015 = Prep4Train(
    train_df=train_df_2015,
    img_dir=IMAGE_2015_DIR,
    img_suffix='.jpeg',
    prep_algo_per_img=preprocessing_algorithm,
    prep_algo_kw={
        'interpolation': cv2.INTER_LANCZOS4,
    },
    cache_dir=CACHE_2015_DIR,
    arcname=ARCNAME_2015,
)

prep2015.before_prep()
print("==== now preprocessing for train-2015...")
tic = datetime.now()
prep2015.prep_using_mp()
toc = datetime.now(); print("preprocess for train spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))
prep2015.post_prep()
prep2015.show_samples_with_comparison()

# ______________________________________________________________________________
# #### for 2019:
prep2019 = Prep4Train(
    train_df=pd.read_csv(CSV_2019_PATH),
    img_dir=IMAGE_2019_DIR,
    img_suffix='.png',
    prep_algo_per_img=preprocessing_algorithm,
    prep_algo_kw={
        'interpolation': cv2.INTER_LANCZOS4,
    },
    cache_dir=CACHE_2019_DIR,
    arcname=ARCNAME_2019,
)

prep2019.before_prep()
print("==== now preprocessing for train-2019...")
tic = datetime.now()
prep2019.prep_using_mp()
toc = datetime.now(); print("preprocess for train spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))
prep2019.post_prep()
prep2019.show_samples_with_comparison()




