#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import json
from datetime import datetime
from fnmatch import filter
from os import path

import pandas as pd
import pytz

from aptos_utils import *

# ______________________________________________________________________________
# Config:
# #### set by user:
DATASET_DIR = '../input/aptos2019-blindness-detection'
# #### auto-generated or doesn't need to change:
MODEL_DIR = path.join('..', 'input', eval('MODEL_NAME'))
MODEL_PAT = '*.h5'
NUM_CLASSES = 5
OUT_CONFIG_PATH = './config.json'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

config = eval('config')

IMAGE_SHAPE = config['preprocess']['image_shape']
# KAPPA_COEF = config['train']['kappa_coef']
# INIT_KAPPA_COEF = config['train']['init_kappa_coef']


# In[]:
# ______________________________________________________________________________
# Main Process
# ####
prep_algo = config['preprocess']['prep_algo']
exec(prep_algo['src'])
handler = eval(prep_algo['func_name'])

test_df = pd.read_csv(path.join(DATASET_DIR, 'test.csv'))

test_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'test_images'], test_df['id_code'], '.png'),
    batch_size=32,
    prep_algo=handler,
    prep_kw=prep_algo['kw'],
    # rescale=1.0 / 255,
    # is_shuffle=True,
)

model_names = sorted(filter(os.listdir(MODEL_DIR), MODEL_PAT))
y_reg_cv = np.zeros([len(model_names), len(test_df), NUM_CLASSES])

for i_model, model_name in enumerate(model_names):
    print(f"Loading {model_name}...")
    model: keras.Model = keras.models.load_model(
        path.join(MODEL_DIR, model_name),
        custom_objects={'crossentropy_with_2logits': crossentropy_with_2logits},
    )
    if i_model == 0:
        model.summary()
    print("==== Now Predict in batches...")
    y_reg_cv[i_model, ...] = model.predict_generator(generator=test_gen, verbose=1)

y_tmp = y_reg_cv.mean(axis=0)
y_pred = np.argmax(y_tmp, axis=-1)

# np.savetxt('y_reg_cv.txt', y_reg_cv.T, delimiter=',')
# y_tmp = KappaOptimizer.regression_2_category(y_reg_cv, coef=INIT_KAPPA_COEF)
np.savetxt('y_tmp.txt', y_tmp.T, delimiter=',')

# #### Submit.
test_df['diagnosis'] = y_pred
test_df['diagnosis'].hist()
print(test_df['diagnosis'].value_counts().sort_index())
test_df.to_csv('submission.csv', index=False)

# #### Dump config.
with open('config.json', 'w') as f:
    config['inference'] = {'create_time': str(datetime.now(tz=REMOTE_TZ)),}
    json.dump(config, f)



