#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"
from os import path

import pandas as pd
from imgaug.parameters import *
from keras_preprocessing.image import ImageDataGenerator

from aptos_utils import *

# ______________________________________________________________________________
# config
DATASET_DIR = '../input/aptos2019-blindness-detection'
IMAGE_SHAPE = [224, 224, 3]
# auto-generated or doesn't need to change:
NUM_CLASSES = 5

train_df = pd.read_csv(path.join(DATASET_DIR, 'train.csv'))

# ______________________________________________________________________________
# Preprocessing Class:


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
    # image = _clahe_rgb(image, clip_limit=2, tile_grid_size=(4,4))
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)# , cv2.INTER_LANCZOS4

    return image


train_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'train_images'], train_df['id_code'], '.png'),
    batch_size=16,
    labels=train_df['diagnosis'],
    prep_algo=preprocessing_algorithm,
    prep_kw={'interpolation': cv2.INTER_LANCZOS4},
    # seqs=[aug_seq(), ],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)

step = 12
img_array, labels = train_gen[step]
show_images_given_array(img_array.astype(int), titles=labels, rows=4, cols=4)

# ______________________________________________________________________________
# aug images:
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),
        # crop images by -5% to 10% of their height/width
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            rotate=(-180, 180),  # rotate by -45 to +45 degrees
            translate_percent={"x": Clip(Normal(0, 0.01), minval=-0.05, maxval=0.05),
                               "y": Clip(Normal(0, 0.01), minval=-0.05, maxval=0.05)}, # translate by -20 to +20 percent (per axis)
            shear=Clip(Normal(0, 5), minval=-10, maxval=10), # shear by -16 to +16 degrees
            order=3, # use nearest neighbour or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval between 0 and 255
            mode='constant', # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            backend='cv2',
        ),
        iaa.CropAndPad(
            percent=Clip(Normal(0, 0.1), minval=-0.2, maxval=0.2),
            pad_mode='constant',
            pad_cval=0,
            sample_independently=False,
        ),
        iaa.Multiply(Clip(Normal(1, 0.1), minval=0.6, maxval=1)),
        # iaa.GammaContrast(Clip(Normal(1, 0.1), minval=0.7, maxval=1.3)),
    ],
    random_order=False
)

aug_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'train_images'], train_df['id_code'], '.png'),
    batch_size=16,
    prep_algo=preprocessing_algorithm,
    prep_kw={'interpolation': cv2.INTER_LANCZOS4},
    seqs=[seq, ],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)

step = 8
img_array, _ = train_gen[step]
show_images_given_array(img_array.astype(int), rows=4, cols=4)

img_array, _ = aug_gen[step]
show_images_given_array(img_array.astype(int), rows=4, cols=4)


# ______________________________________________________________________________
# Preprocessed images:
PREPROCESSED_DIR = '../tmp/aptos-preprocessed'

preprocessed_gen = MyGenerator(
    img_paths=pathj([PREPROCESSED_DIR, 'train_images'], train_df['id_code'], '.png'),
    batch_size=0x100,
    labels=train_df['id_code'],
    # prep_algo=preprocessing_algorithm,
    # prep_kw={'interpolation': cv2.INTER_LANCZOS4},
    # seqs=[aug_seq(), ],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)

def m_func(batch):
    bat_norm = np.max(batch[0], axis=(1,2,3))
    return pd.DataFrame({'id_code': batch[1], 'norm': bat_norm})

norm_table = pd.concat(list(map(m_func, preprocessed_gen)))

lt_pd = norm_table[norm_table['norm'] < 200]
lt_paths = pathj([PREPROCESSED_DIR, 'train_images'], lt_pd['id_code'], '.png')

# #### batch_step
step=3
s, e = np.array([0,16]) + step * 16
show_images_given_paths(
    paths=lt_paths[s:e], rows=4, cols=4
)

# ______________________________________________________________________________
# Compare:
comp_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'train_images'], lt_pd['id_code'], '.png'),
    batch_size=16,
    prep_algo=preprocessing_algorithm,
    prep_kw={'interpolation': cv2.INTER_LANCZOS4},
    # seqs=[aug_seq(), ],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)
img_array, _ = comp_gen[step]
show_images_given_array(img_array.astype(int), rows=4, cols=4)


# ______________________________________________________________________________
#
gen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    # brightness_range=(1, 2),
    # shear_range=90,
    zoom_range=0.,
    channel_shift_range=200,
    fill_mode='constant',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    # rescale=1/255,
    preprocessing_function=None,
    data_format='channels_last',
    validation_split=0.0,
    dtype='float32'
)

train_df['img_name'] = train_df['id_code'] + '.png'
img_iter = gen.flow_from_dataframe(
    dataframe=train_df,
    directory=path.join(DATASET_DIR, 'train_images'),
    x_col="img_name", y_col="diagnosis", has_ext=True,
    target_size=(224, 224), color_mode='rgb',
    classes=None, class_mode='other',
    batch_size=32, shuffle=True, seed=2015,
    interpolation='nearest'
)

x, y = img_iter.next()





