#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import hashlib
import inspect
import json
import shutil
import tarfile
from datetime import datetime
from os import path

import pandas as pd
import pytz
from pandas.core.groupby import DataFrameGroupBy

from aptos_utils import *

# ______________________________________________________________________________
# Config:
# #### set by user:
DATASET_DIR = '../input/aptos2019-blindness-detection'
IMAGE_SHAPE = [380, 380, 3]
OUT_DIR = './'
ARCNAME = 'aptos-preprocessed'
# #### auto-generated or doesn't need to change:
NUM_CLASSES = 5
CACHE_DIR = '/tmp/aptos-preprocessed'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

config = {}

# ______________________________________________________________________________
# de-duplication:
train_df = pd.read_csv(path.join(DATASET_DIR, 'train.csv'))


def img_id_to_path(id: str, sub_dir:str='train_images', dir:str=DATASET_DIR):
    return path.join(dir, sub_dir, id + '.png')


def get_md5_digest(file_path):
    with open(file_path, 'rb') as f:
        m = hashlib.md5()
        while True:
            str_read = f.read(0x1000)
            if len(str_read) == 0:
                break
            m.update(str_read)
        return m.hexdigest()


# Calc MD5.
with Pool() as p:
    md5_digest_list = list(tqdm(p.imap(
        get_md5_digest,
        [img_id_to_path(id) for id in train_df['id_code']],
        chunksize = 100,
    )))
train_df['md5'] = md5_digest_list
print(train_df)

gb: DataFrameGroupBy = train_df.groupby('md5')

#
dup_dict = {}
for key, group in gb:
    dup_degree = len(group)
    if dup_dict.get(dup_degree) is None:
        dup_dict[dup_degree] = []
    dup_dict[dup_degree].append(group)
# duplication summary.
for deg, group_list in dup_dict.items():
    print(f'dup degree: {deg}; num groups: {len(group_list)}')


def plot_dup(df: pd.DataFrame):
    print('==== i = a = m = s = p = l = i = t = e = r ===='*2)
    cols = 2
    rows = ceil(len(df) / cols)
    fig = plt.figure(figsize = (8*cols, 6*rows))
    for i, t in enumerate(df.itertuples()):
        img_path = img_id_to_path(t.id_code)
        sp = fig.add_subplot(rows, cols, i + 1)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sp.set_title(t.id_code + f' Label: {t.diagnosis}')
        sp.imshow(img)
    plt.show()


# visual duplicated images.
for df in dup_dict[2]:
    plot_dup(df)
for df in dup_dict[3]:
    plot_dup(df)
for df in dup_dict[4]:
    plot_dup(df)

# deduplication algorithm.
frames = []
eye = np.eye(NUM_CLASSES, dtype=int)
for key, group in gb:
    df = pd.DataFrame()
    df['id_code'] = group.head(1)['id_code']
    labels = np.array(group['diagnosis'])
    # diagnosis field's shape: (NUM_CLASSES, ), index i's value means: class i's counts.
    df['diagnosis'] = [list(eye[:, labels].sum(axis=-1))]
    df['md5'] = key
    frames.append(df)
dedup_train_df = pd.concat(frames).sort_index()

# ______________________________________________________________________________
# Preprocessing Class:


class Prep4Train(PreprocessBase):

    def __init__(self, dataset_dir, prep_algo_per_img, prep_algo_kw=None, train_df=None,
                 img_per_chunk=0x100, out_dir=OUT_DIR, arcname=ARCNAME, cache_dir=CACHE_DIR):
        self.train_df = train_df if train_df is not None else pd.read_csv(path.join(dataset_dir, 'train.csv'))
        super().__init__(self.train_df['id_code'], img_per_chunk)
        #
        self.dataset_dir = dataset_dir
        self.img_dir = path.join(dataset_dir, 'train_images')
        self.prep_algo_per_img = prep_algo_per_img
        self.prep_algo_kw = prep_algo_kw if prep_algo_kw is not None else {}
        self.out_dir = out_dir
        self.arcname = arcname
        self.cache_dir = cache_dir

    def prep_one_img(self, img_id):
        img_path = path.join(self.img_dir, img_id + '.png')
        out_path = path.join(self.cache_dir, 'train_images', img_id + '.png')
        # apply preprocessing algo, convert color & save.
        img = self.prep_algo_per_img(img_path, **self.prep_algo_kw)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img)

    def before_prep(self):
        if not path.exists(path.join(self.cache_dir, 'train_images')):
            os.makedirs(path.join(self.cache_dir, 'train_images'), exist_ok=True)

    def post_prep(self):
        # copy .csv file to cache dir for later use.
        for csv_name in [x for x in os.listdir(self.dataset_dir) if path.splitext(x)[1] == '.csv']:
            if csv_name != 'train.csv':
                shutil.copyfile(path.join(self.dataset_dir, csv_name),
                                path.join(self.cache_dir, csv_name))
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


# In[]:
# ______________________________________________________________________________
# main process:

# ______________________________________________________________________________
# #### for train:

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
    # cv2.normalize(src=image, dst=image, alpha=255, norm_type=cv2.NORM_INF)
    image = cv2.resize(image, tuple(IMAGE_SHAPE[:-1]), interpolation)
    image = _clahe_rgb(image, clip_limit=5, tile_grid_size=(4,4))
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)# , cv2.INTER_LANCZOS4

    return image


prep = Prep4Train(
    dataset_dir=DATASET_DIR,
    prep_algo_per_img=preprocessing_algorithm,
    prep_algo_kw={
        'interpolation': cv2.INTER_LANCZOS4,
    },
    train_df=dedup_train_df,
)

prep.before_prep()
print("==== now preprocessing for train...")
tic = datetime.now()
prep.prep_using_mp()
toc = datetime.now(); print("preprocess for train spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))
prep.post_prep()


# In[]:
# ______________________________________________________________________________
# #### for test:

del prep.prep_algo_per_img


class Prep4Test(PreprocessBase):

    def __init__(self, dataset_dir, prep_algo_per_img, prep_algo_kw=None, img_per_chunk=100):
        self.test_df = pd.read_csv(path.join(dataset_dir, 'test.csv'))
        img_ids = self.test_df['id_code']
        super().__init__(img_ids, img_per_chunk)
        self.img_dir = path.join(dataset_dir, 'test_images')
        self.prep_algo_per_img = prep_algo_per_img
        self.prep_algo_kw = prep_algo_kw if prep_algo_kw is not None else {}

    def prep_one_img(self, img_id):
        img_path = path.join(self.img_dir, img_id + '.png')
        return self.prep_algo_per_img(img_path, **self.prep_algo_kw)


with open(path.join(OUT_DIR, 'config.json'), 'r') as f:
    prep_algo = json.load(f)['preprocess']['prep_algo']
    exec(prep_algo['src'])
    handler = eval(prep_algo['func_name'])

prep = Prep4Test(
    dataset_dir=DATASET_DIR,
    prep_algo_per_img=handler,
    prep_algo_kw=prep_algo['kw'],
)

print("==== now preprocessing for test...")
tic = datetime.now()
x_test = np.array(prep.prep_using_mp())
toc = datetime.now(); print("preprocess for test spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))
print("x_test shape: " + str(x_test.shape))



