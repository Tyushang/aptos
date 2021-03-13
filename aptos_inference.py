#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import json
from datetime import datetime
from os import path

import pandas as pd
import pytz

from aptos_utils import *

# ______________________________________________________________________________
# Config:
# #### set by user:
TTA_ROUNDS = 7
DATASET_DIR = '../input/aptos2019-blindness-detection'
# #### auto-generated or doesn't need to change:
MODEL_PATH = path.join('..', 'input', eval('MODEL_NAME'), 'model.h5')
OUT_CONFIG_PATH = './config.json'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

config = eval('config')

IMAGE_SHAPE = config['preprocess']['image_shape']
KAPPA_COEF = config['train']['kappa_coef']
INIT_KAPPA_COEF = config['train']['init_kappa_coef']


# In[]:
# ______________________________________________________________________________
# Main Process
# #### loading model...
print("loading model...")
model: keras.Model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={'relu_max5': relu_max5},
)
model.summary()

# #### Predict and Submit.
prep_algo = config['preprocess']['prep_algo']
exec(prep_algo['src'])
handler = eval(prep_algo['func_name'])

test_df = pd.read_csv(path.join(DATASET_DIR, 'test.csv'))

seq_tta = iaa.Sequential(
    [
        # crop images by -5% to 10% of their height/width
        iaa.CropAndPad(
            percent=(0, 0.1),
            pad_mode='constant',
            pad_cval=0,
            sample_independently=False,
        ),
        # apply the following augmenters to most images
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            rotate=(-5, 5),  # rotate by -45 to +45 degrees
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
            # shear=(-16, 16), # shear by -16 to +16 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval between 0 and 255
            mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
        iaa.ContrastNormalization((0.9, 1.2)),
        iaa.Multiply((0.9, 1.2)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
    ],
    random_order=False
)

test_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'test_images'], test_df['id_code'], '.png'),
    batch_size=32,
    prep_algo=handler,
    prep_kw=prep_algo['kw'],
    seqs=[seq_tta, ],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)

print("==== Now Predict in batches...")
y_tta = np.zeros([TTA_ROUNDS, len(test_df)])
for i_tta in range(TTA_ROUNDS):
    y_tta[i_tta, :] = model.predict_generator(generator=test_gen, verbose=1).flatten()
y_evidence = KappaOptimizer.regression_2_category(y_tta, coef=INIT_KAPPA_COEF)


def vote(x, vote_axis=0):
    """ return most frequent number along vote_axis. """
    one_hot = x[..., np.newaxis] == np.array([0,1,2,3,4])
    distribution: np.ndarray = one_hot.sum(axis=vote_axis)
    return distribution.argmax(axis=-1)


y_pred = vote(y_evidence, vote_axis=0)

# #### Submit.
test_df['diagnosis'] = y_pred
test_df['diagnosis'].hist()
print(test_df['diagnosis'].value_counts().sort_index())
test_df.to_csv('submission.csv', index=False)

# #### Dump config.
with open('config.json', 'w') as f:
    config['inference'] = {'create_time': str(datetime.now(tz=REMOTE_TZ)),}
    json.dump(config, f)



