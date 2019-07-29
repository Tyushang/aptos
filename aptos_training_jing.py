# coding: utf-8

# In[ ]:


import json
import tarfile
from datetime import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy as sp
from PIL import Image
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# get_ipython().run_line_magic('matplotlib', 'inline')

# ______________________________________________________________________________
# Config:
# set by user:
PREPROCESSED_TAR_FILE = '../input/aptos-preprocessing-ben/aptos-preprocessed.tar'
ARCNAME = 'aptos-preprocessed'
NUM_EPOCHS = 30
BATCH_SIZE = 32
# auto-generated or doesn't need to change:
NUM_CLASSES = 5
MODEL_PATH = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

with tarfile.open(PREPROCESSED_TAR_FILE, 'r') as tar:
    config = eval(tar.extractfile(f'{ARCNAME}/config.json').read())

IMAGE_SHAPE = config['image_shape']


# In[ ]:

def display_samples(tar, df, columns=4, rows=3):
    fig = plt.figure(figsize=(5 * columns, 4 * rows))

    for i in range(columns * rows):
        img_id = df.loc[i, 'id_code']
        img_label = df.loc[i, 'diagnosis']
        #         tar.extract(f'{ARC_NAME}/train_images/{image_id}.png', f'/tmp')
        #         img = cv2.imread(f'/tmp/{ARC_NAME}/train_images/{image_id}.png')
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(tar.extractfile(f'{ARCNAME}/train_images/{img_id}.png'))

        fig.add_subplot(rows, columns, i + 1)
        plt.title(img_label)
        plt.imshow(img)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# In[ ]:

with tarfile.open(PREPROCESSED_TAR_FILE, 'r') as tar:
    # ## Extract train.csv, test.csv ----> train_df, test_df
    train_val_df = pd.read_csv(tar.extractfile(f'{ARCNAME}/train.csv'))
    test_df = pd.read_csv(tar.extractfile(f'{ARCNAME}/test.csv'))
    # ## Display train_df, test_df info.
    print(train_val_df.shape)
    print(test_df.shape)
    train_val_df.head()
    train_val_df['diagnosis'].hist()
    train_val_df['diagnosis'].value_counts()
    # ## Display sample images.
    display_samples(tar, train_val_df)

    # ## Extract train_images/*.png ----> x_train_val
#     N = train_val_df.shape[0]
#     x_train_val = np.empty((N, *IMAGE_SHAPE), dtype=np.uint8)
#     for i, img_id in enumerate(tqdm(train_val_df['id_code'])):
#         file_obj = tar.extractfile(f'{ARCNAME}/train_images/{img_id}.png')
#         x_train_val[i, :, :, :] = Image.open(file_obj)
    x_train_val = np.array(list(tqdm(map(
        lambda id: np.array(Image.open(tar.extractfile(f'{ARCNAME}/train_images/{id}.png'))),
        train_val_df['id_code']
    ))))

# In[ ]:

y_train_val = train_val_df['diagnosis']

print(x_train_val.shape)
print(y_train_val.shape)

# In[ ]:

x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val,
    test_size=0.15,
    # stratify=y_train_val,
    random_state=2019
)

# In[ ]:


class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


# In[ ]:


def create_datagen():
    return ImageDataGenerator(
#         samplewise_center=True,
#         samplewise_std_normalization=True,
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )


# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE)
# Using Mixup
mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()

data_generator.next()

# In[ ]:

# refer: https://www.kaggle.com/hmendonca/aptos19-regressor-fastai-oversampling-tta
# and refactor it.
class KappaOptimizer():
    def __init__(self, coef=(0.5, 1.5, 2.5, 3.5)):
        self.coef = list(coef)
        # define score function:
        self.func = self.quad_kappa

    @classmethod
    def _predict(cls, coef, reg):
        # using numpy broadcasting feature to simplify code.
        # see: https://docs.scipy.org/doc/numpy-1.13.0/reference/ufuncs.html#ufuncs-broadcasting
        return (reg.reshape([-1, 1]) > np.array(coef).reshape([1, -1])).sum(axis=1)

    @classmethod
    def _quad_kappa(cls, coef, reg, y):
        y_hat = cls._predict(coef, reg)
        return cohen_kappa_score(y, y_hat, weights='quadratic')

    def predict(self, preds):
        return self._predict(self.coef, preds)

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


kappa_opt = KappaOptimizer()


# In[ ]:

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]

        y_pred_regression = self.model.predict(X_val)
        y_pred = kappa_opt.predict(y_pred_regression)

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


tbCallBack = TensorBoard(log_dir="./tensorboard_log", histogram_freq=2)

# # Model: DenseNet-121

# In[ ]:

densenet = DenseNet121(
    weights=MODEL_PATH,
    include_top=False,
    input_shape=IMAGE_SHAPE
)

# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='sigmoid'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(128, activation='softplus'))
    # # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='relu'))

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )

    return model


# In[ ]:

model = build_model()
model.summary()

# # Training & Evaluation

# In[ ]:

kappa_metrics = Metrics()

tic = datetime.now()
history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[
        # Interrupt training if `val_loss` stops improving for over $patience epochs
        keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        # Track kappa,
        kappa_metrics,
        # Tensor Board log.
        tbCallBack,
    ]
)
toc = datetime.now(); print("model.fit_generator spend: %f s" % (toc - tic).total_seconds())

# In[ ]:

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

# In[ ]:

plt.plot(kappa_metrics.val_kappas)

# ______________________________________________________________________________
# fit kappa and post processing.
print("loading model...")
model = keras.models.load_model('model.h5')
model.summary()

tic = datetime.now()
y_regression = model.predict(x_train_val)
toc = datetime.now(); print("model.predict spend: %f s" % (toc - tic).total_seconds())

kappa_opt.fit(y_regression, y_train_val)

# dump config for later use.
config['kappa_coef'] = kappa_opt.coef
config['create_time'] = str(datetime.now(tz=REMOTE_TZ))
with open('./config.json', 'w') as f:
    json.dump(config, f)

# copy preprocessing algorithm source code for later use.
with tarfile.open(PREPROCESSED_TAR_FILE, 'r') as tar:
    tar.extract(f'{ARCNAME}/prep_algo.py')




