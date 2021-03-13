# %% [markdown]
# this kernel is from
# https://www.kaggle.com/manojprabhaakr/similar-duplicate-images-in-aptos-data
# and  https://www.kaggle.com/maxwell110/duplicated-list-csv-file/
#
# I do three things:
# 1. change phash to md5 according see-'s comment https://www.kaggle.com/maxwell110/duplicated-list-csv-file/comments#575422;
# 2. Duplicated with different label
# 3. Duplicated in both train and test.

# %% [code]
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))
import sys;
import hashlib;
from os.path import isfile
from joblib import Parallel, delayed
import psutil

# %% [code]
train_df = pd.read_csv("../input/train.csv")
print(train_df.shape)
test_df = pd.read_csv("../input/sample_submission.csv")
test_df['diagnosis'] = np.nan
train = train_df.append(test_df)


# %% [code]
def expand_path(p):
    if isfile('../input/train_images/' + p + '.png'): return '../input/train_images/' + p + '.png'
    if isfile('../input/test_images/' + p + '.png'): return '../input/test_images/' + p + '.png'
    return p


def getImageMetaData(p):
    strFile = expand_path(p)
    file = None;
    bRet = False;
    strMd5 = "";

    try:
        file = open(strFile, "rb");
        md5 = hashlib.md5();
        strRead = "";

        while True:
            strRead = file.read(8096);
            if not strRead:
                break;
            md5.update(strRead);
        # read file finish
        bRet = True;
        strMd5 = md5.hexdigest();
    except:
        bRet = False;
    finally:
        if file:
            file.close()

    return p, strMd5


# %% [code]
img_meta_l = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(
    (delayed(getImageMetaData)(fp) for fp in train.id_code))

# %% [code]
img_meta_df = pd.DataFrame(np.array(img_meta_l))
img_meta_df.columns = ['id_code', 'strMd5']

# %% [code]
train = train.merge(img_meta_df, on='id_code')

# %% [code]
train['strMd5_count'] = train.groupby('strMd5').id_code.transform('count')

# %% [code]
train['strMd5_train_count'] = train['strMd5'].map(
    train.groupby('strMd5')['diagnosis'].apply(lambda x: x.notnull().sum()))

# %% [code]
train['strMd5_nunique'] = train.groupby('strMd5')['diagnosis'].transform('nunique').astype('int')

# %% [code]
train.to_csv('strMd5.csv', index=None)

# %% [code]
train[train.strMd5_count > 1].strMd5_count.value_counts().sort_index()

# %% [code]
import matplotlib.pyplot as plt
import cv2

# %% [markdown]
# **Duplicated with same label**

# %% [code]
train[(train.strMd5_train_count > 1) & (train.strMd5_nunique == 1)].strMd5_count.value_counts()

# %% [code]
strMd51 = train[(train.strMd5_count > 1) & (train.strMd5_nunique == 1)].strMd5.unique()
strMd5 = strMd51[0]
size = len(train[train['strMd5'] == strMd5]['id_code'])
fig = plt.figure(figsize=(20, 5))
for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):
    y = fig.add_subplot(1, size, idx + 1)
    img = cv2.imread(expand_path(img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_id = train[train.id_code == img_name]['diagnosis'].values
    y.set_title(img_name + f'Label: {class_id}')
    y.imshow(img)
plt.show()

# %% [markdown]
# **Duplicated with different label**

# %% [code]
train[(train.strMd5_count > 1) & (train.strMd5_nunique > 1)].strMd5_count.value_counts()

# %% [code]
strMd52 = train[(train.strMd5_count > 1) & (train.strMd5_nunique > 1)].strMd5.unique()
strMd5 = strMd52[0]
for strMd5 in strMd52[:5]:
    size = len(train[train['strMd5'] == strMd5]['id_code'])
    fig = plt.figure(figsize=(20, 5))
    for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):
        y = fig.add_subplot(1, size, idx + 1)
        img = cv2.imread(expand_path(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_id = train[train.id_code == img_name]['diagnosis'].values
        y.set_title(img_name + f'Label: {class_id}')
        y.imshow(img)
    plt.show()

# %% [markdown]
# **Duplicated in both train and test**

# %% [code]
train[(train.strMd5_count > 1) & (train.diagnosis.isnull())].shape[0]

# %% [code]
strMd52 = train[(train.strMd5_count > 1) & (train.diagnosis.isnull())].strMd5.unique()
strMd5 = strMd52[0]
for strMd5 in strMd52[:5]:
    size = len(train[train['strMd5'] == strMd5]['id_code'])
    fig = plt.figure(figsize=(20, 5))
    for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):
        y = fig.add_subplot(1, size, idx + 1)
        img = cv2.imread(expand_path(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_id = train[train.id_code == img_name]['diagnosis'].values
        y.set_title(img_name + f'Label: {class_id}')
        y.imshow(img)
    plt.show()

# %% [markdown]
# **About leak**

# %% [code]
train[(train.strMd5_count == 2)]['strMd5_train_count'].value_counts().sort_index()

# %% [code]
strMd52 = train[(train.strMd5_count > 2)].strMd5.unique()
strMd5 = strMd52[0]
for strMd5 in strMd52:
    size = len(train[train['strMd5'] == strMd5]['id_code'])
    fig = plt.figure(figsize=(20, 5))
    for idx, img_name in enumerate(train[train['strMd5'] == strMd5]['id_code'][:size]):
        y = fig.add_subplot(1, size, idx + 1)
        img = cv2.imread(expand_path(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_id = train[train.id_code == img_name]['diagnosis'].values
        y.set_title(img_name + f'Label: {class_id}')
        y.imshow(img)
    plt.show()

# %% [markdown]
# **conclusion**
# there are 2×255 2-duplicated image.
# 2×116 are in train. 89 have same label, 27 have differnet label;
# 2×134 are in train and test,which means leak;
# 2×5 are in test.
#
# there are 3×3 3-duplicated image.
# 1×3 are in train.
# 2×3 are in train and test.
#
# there are 3×4 4-duplicated image.
# 3×4 are in train and test.
#
# there are 1×5 5-duplicated image.
# 1×5 are in train and test.
#
# there are total 141(134+7) leak in test.

# %% [code]
