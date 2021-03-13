#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__ = u"Frank Jing"

import abc
import os
import queue
from math import ceil
from multiprocessing.pool import Pool
from typing import *

import imgaug.augmenters as iaa
import keras
import numpy as np
import pandas as pd
import scipy as sp
# noinspection PyUnresolvedReferences
from PIL import Image
# noinspection PyUnresolvedReferences
from cv2 import cv2
from matplotlib import pyplot as plt
from scipy.linalg import circulant
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import shuffle
from tqdm import tqdm


# ______________________________________________________________________________
# Class:


class PreprocessBase():

    def __init__(self, img_ids, img_per_chunk=100):
        self.img_ids = img_ids
        self.chunk_size = img_per_chunk

    def prep_using_mp(self):
        with Pool() as pool:
            res = list(tqdm(pool.imap(self.prep_one_img, self.img_ids, self.chunk_size)))
        return res

    def prep_datagen_with_len(self, batch_size=0x100):
        steps = ceil(len(self.img_ids) / batch_size)

        def _datagen():
            for i_step in range(steps):
                start_tmp, end_tmp = tuple(np.array([0, batch_size]) + i_step*batch_size)
                sub_ids = self.img_ids[start_tmp : end_tmp]
                yield np.stack(list(map(self.prep_one_img, sub_ids)))

        return _datagen(), steps

    @abc.abstractmethod
    def prep_one_img(self, img_id): ...


class KappaOptimizer():
    # refer: https://www.kaggle.com/hmendonca/aptos19-regressor-fastai-oversampling-tta
    # and refactor it.
    def __init__(self, coef=(0.5, 1.5, 2.5, 3.5)):
        self.coef = list(coef)
        # define score function:
        self.func = self.quad_kappa

    @classmethod
    def regression_2_category(cls, reg, coef=(0.5, 1.5, 2.5, 3.5)):
        # using numpy broadcasting feature to simplify code.
        # see: https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#ufuncs-broadcasting
        relation_table: np.ndarray = np.array(reg)[..., np.newaxis] > np.array(coef)
        return relation_table.sum(axis=-1)

    @classmethod
    def _quad_kappa(cls, coef, reg, y):
        y_hat = cls.regression_2_category(reg, coef)
        return cohen_kappa_score(y, y_hat, weights='quadratic')

    def predict(self, preds):
        return self.regression_2_category(preds, self.coef)

    def quad_kappa(self, regression_preds, y):
        return self._quad_kappa(self.coef, regression_preds, y)

    def fit(self, regression_preds, y):
        ''' maximize quad_kappa '''
        print('Early score:', self.quad_kappa(regression_preds, y))
        neg_kappa = lambda coef: -self._quad_kappa(coef, regression_preds, y)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.coef, method='nelder-mead',
                                       options={'maxiter': 1000, 'fatol': 1e-20, 'xatol': 1e-20})
        print(opt_res)
        self.coef = opt_res.x.tolist()
        print('New score:', self.quad_kappa(regression_preds, y))

    # def forward(self, preds, y):
    #     ''' the pytorch loss function '''
    #     return torch.tensor(self.quad_kappa(preds, y))


class MyGenerator(keras.utils.Sequence):

    def __init__(self, img_paths: List[str], batch_size: int,
                 labels=None,
                 prep_algo: Callable[[np.ndarray], np.ndarray] = None,
                 prep_kw: dict = None,
                 seqs: List[iaa.Sequential] = None,
                 rescale=1.0,
                 is_shuffle=False,
                 is_mixup=False):
        self.img_paths: List[str] = img_paths
        self.batch_size: int = batch_size
        self.labels = labels if labels is not None else np.zeros([len(img_paths), ])
        self.prep_algo: Callable[[np.ndarray], np.ndarray] = prep_algo
        self.prep_algo_kw: dict = prep_kw if prep_kw is not None else {}
        self.seqs: List[iaa.Sequential] = seqs if seqs is not None else []
        self.rescale = rescale
        self.is_shuffle = is_shuffle
        self.is_mixup = is_mixup
        if self.is_shuffle:
            self._shuffle_paths_labels()

    def __len__(self):
        return ceil(len(self.img_paths) / float(self.batch_size))

    def __getitem__(self, idx):
        start, end  = np.array([0, self.batch_size]) + self.batch_size * idx
        batch_paths = self.img_paths[start : end]
        batch_y = self.labels[start : end]

        batch_x = self._get_images(batch_paths)
        if self.is_mixup:
            batch_x, batch_y = self._mixup(batch_x, batch_y)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.is_shuffle:
            self._shuffle_paths_labels()

    @staticmethod
    def _mixup(x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        return mixed_x, mixed_y

    def _get_images(self, img_paths: List[str]) -> np.ndarray:
        # do preprocessing if image files were original. else read them directly(they were preprocessed.).
        if self.prep_algo is not None:
            lam = lambda p: self.prep_algo(p, **self.prep_algo_kw)
        else:
            lam = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        images = np.array(list(map(lam, img_paths)))

        for seq in self.seqs:
            images = seq.augment_images(images)

        return images * self.rescale

    def _shuffle_paths_labels(self):
        self.img_paths, self.labels = shuffle(self.img_paths, self.labels)


# class AdaptiveGenerator(MyGenerator):
#     LABEL = [0, 1, 2, 3, 4]
#
#     def __init__(self, img_paths: List[str], batch_size: int, y_valid_true: List[int],
#                  labels: List[int] = None,
#                  prep_algo: Callable[[np.ndarray], np.ndarray] = None,
#                  prep_kw: dict = None,
#                  seqs: List[iaa.Sequential] = None,
#                  rescale=1.0,
#                  is_shuffle=False):
#         super().__init__(img_paths, batch_size, labels, prep_algo, prep_kw,
#                          seqs, rescale, is_shuffle, is_mixup=False)
#         # copy origin img_paths and labels.
#         self.origin_img_paths = img_paths
#         self.origin_labels = labels
#         self.num_samples = len(img_paths)
#         self.group_by_label = pd.DataFrame({'img': img_paths, 'label': labels}).groupby('label')
#         # determine next epoch is adaptive or origin.
#         self.is_next_epoch_adaptive = True
#         # queue to receive y_valid_pred
#         self.queue = queue.Queue()
#         self.y_valid_true = y_valid_true
#
#     def on_epoch_end(self):
#         y_valid_pred = self.queue.get()
#         # origin epoch and adaptive epoch runs alternately.
#         if self.is_next_epoch_adaptive:
#             self.img_paths, self.labels = self._resample(y_valid_pred)
#         else:
#             self.img_paths, self.labels = self.origin_img_paths, self.origin_labels
#         # shuffle if required.
#         if self.is_shuffle:
#             self._shuffle_paths_labels()
#         self.is_next_epoch_adaptive = not self.is_next_epoch_adaptive
#
#         self.queue.task_done()
#
#     def _resample(self, y_valid_pred):
#         """resample images by prob."""
#         # resample probability for every label.
#         resample_prob = self._get_resample_prob(y_valid_pred)
#         # resample times for every label.
#         label_freq = self._prob_2_freq(resample_prob)
#         # do resample according to label_freq.
#         frames = []
#         for label, group_df in self.group_by_label:
#             frames += group_df.sample(label_freq[label], replace=True)
#         resampled_df: pd.DataFrame = shuffle(pd.concat(frames))
#
#         return list(resampled_df['img']), list(resampled_df['label'])
#
#     def _get_resample_prob(self, y_valid_pred):
#         """if predict is right, nothing to do; else, contribute to resample prob of both label(pred and true)."""
#         idx = self.y_valid_true != y_valid_pred
#         cat = np.concatenate((self.y_valid_true[idx], y_valid_pred[idx]))
#         counts = np.bincount(cat)
#         return counts / counts.sum()
#
#     def _prob_2_freq(self, prob):
#         """same with prob*self.num_samples theoretically."""
#         label_samples = np.random.choice(self.LABEL, self.num_samples, p=prob)
#         return np.bincount(label_samples)


# ______________________________________________________________________________
# functions:


def pathj(dirs: List[str], filenames: Iterable[str], extname='.png'):
    return [os.path.join(*dirs, fn + extname) for fn in filenames]


def show_images_given_paths(paths: list, titles: list=None, rows=3, cols=4):
    img_array = [np.array(Image.open(p)) for p in paths]
    if titles is None:
        titles = [os.path.basename(p) for p in paths]
    _show_samples(np.array(img_array), list(titles), rows, cols)


def show_images_given_array(img_array: np.ndarray, titles: list=None, rows=3, cols=4):
    if titles is None:
        titles = list(range(len(img_array)))
    _show_samples(np.array(img_array), list(titles), rows, cols)


def _show_samples(img_array: np.ndarray, titles: list=None, rows=3, cols=4):
    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    for i, img in enumerate(img_array[:(rows*cols)]):
        fig.add_subplot(rows, cols, i + 1)
        if titles is not None:
            plt.title(str(titles[i]))
        plt.imshow(img)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def label_2_onehot(label: np.ndarray, num_classes: int = 5):
    """encode every element of label using one-hot."""
    eye = np.eye(num_classes, dtype=int)
    return eye[label]


def onehot_2_label(onehot: np.ndarray):
    """decode one-hot to category label."""
    return np.argmax(onehot, axis=-1)


def label_counts_2_logits(label_counts, multiplier=1.0, bias=0.0, logits_of_NoDR = (4,3,2,1,0)):
    """convert label-counts to logits."""
    cir = circulant(np.array(logits_of_NoDR))
    logits_table = np.tril(cir) + np.tril(cir, k=-1).T
    # label_counts' shape: [n_samples, NUM_CLASSES]
    label_counts = np.array(label_counts, dtype=int)
    # return value shape: [n_samples, NUM_CLASSES]
    return multiplier * np.matmul(label_counts, logits_table) + bias


def label_counts_2_regression(label_counts: np.ndarray, regression_table=(0, 1, 2, 3, 4)):
    # label_counts' shape: [n_samples, NUM_CLASSES]
    label_distribution = label_counts / label_counts.sum(axis=-1, keepdims=True)
    return np.matmul(label_distribution, np.array(regression_table))


def category_2_regression(cate: np.ndarray, regression_table=(0,1,2,3,4)):
    return np.array(regression_table)[cate]


def label_2_multi(label: np.ndarray, n_classes=5):
    multi_table = np.tril(np.ones([n_classes, n_classes], dtype=int))
    return multi_table[label]


def multi_2_label(multi: np.ndarray, threshold=0.5):
    return (multi > threshold).sum(axis=-1) - 1


def crossentropy_with_2logits(y_true, y_pred):
    from keras.activations import softmax
    from keras.losses import categorical_crossentropy
    return categorical_crossentropy(softmax(y_true), softmax(y_pred))


def relu_max5(x):
    from keras.activations import relu
    return relu(x, max_value=5.)


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)



