#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import json
from datetime import datetime
from fnmatch import filter
from os import path

import pandas as pd
import pytz
from imgaug.parameters import Normal, Clip

from aptos_utils import *

# ______________________________________________________________________________
# Config:
# #### set by user:
DATASET_DIR = '../input/aptos2019-blindness-detection'
# #### auto-generated or doesn't need to change:
MODEL_DIR = path.join('..', 'input', eval('MODEL_NAME'))
MODEL_PAT = '*.h5'
NUM_CLASSES = 5
TTA_ROUNDS = 3
OUT_CONFIG_PATH = './config.json'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

config = eval('config')

IMAGE_SHAPE = config['preprocess']['image_shape']
INIT_KAPPA_COEF = config['train']['init_kappa_coef']


# In[]:
# ______________________________________________________________________________
# Main Process
# ####
prep_algo = config['preprocess']['prep_algo']
exec(prep_algo['src'])
handler = eval(prep_algo['func_name'])

test_df = pd.read_csv(path.join(DATASET_DIR, 'test.csv'))

aug = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            rotate=Clip(Normal(0, 5), minval=-10, maxval=10),  # rotate by -45 to +45 degrees
            translate_percent={"x": Clip(Normal(0, 0.025), minval=-0.05, maxval=0.05),
                               "y": Clip(Normal(0, 0.025), minval=-0.05, maxval=0.05)}, # translate by -20 to +20 percent (per axis)
            # shear=(-16, 16), # shear by -16 to +16 degrees
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
        iaa.Multiply(Clip(Normal(1, 0.1), minval=0.8, maxval=1)),
        # iaa.GammaContrast(Clip(Normal(1, 0.1), minval=0.75, maxval=1.5)),
    ],
    random_order=False
)

test_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'test_images'], test_df['id_code'], '.png'),
    batch_size=32,
    prep_algo=handler,
    prep_kw=prep_algo['kw'],
    seqs=[aug, ],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)

model_names = sorted(filter(os.listdir(MODEL_DIR), MODEL_PAT))
y_reg_cv_tta = np.zeros([len(model_names), TTA_ROUNDS, len(test_df)])

for i_model, model_name in enumerate(model_names):
    print(f"Loading {model_name}...")
    model: keras.Model = keras.models.load_model(
        path.join(MODEL_DIR, model_name),
        custom_objects={'relu_max5': relu_max5},
    )
    if i_model == 0:
        model.summary()
    for i_tta in range(TTA_ROUNDS):
        print("==== Now Predict in batches...")
        y_reg_cv_tta[i_model, i_tta, ...] = model.predict_generator(generator=test_gen, verbose=1).flatten()

y_ensemble = y_reg_cv_tta.mean(axis=(0, 1))
y_pred = KappaOptimizer.regression_2_category(y_ensemble, coef=INIT_KAPPA_COEF)

np.savetxt('y_ensemble.txt', y_ensemble.T, delimiter=',')

# #### Submit.
test_df['diagnosis'] = y_pred
test_df['diagnosis'].hist()
print(test_df['diagnosis'].value_counts().sort_index())
test_df.to_csv('submission.csv', index=False)

# #### Dump config.
with open('config.json', 'w') as f:
    config['inference'] = {'create_time': str(datetime.now(tz=REMOTE_TZ)),}
    json.dump(config, f)



