# coding: utf-8

# In[ ]:
import inspect
import json
from datetime import datetime
from os import path

import pandas as pd
import pytz
from efficientnet import EfficientNetB3
from imgaug.parameters import *
from keras import layers, models
from keras.callbacks import Callback, TensorBoard
from keras.optimizers import Adam

from aptos_utils import *

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:
# ______________________________________________________________________________
# Config:
# set by user:
DATASET_DIR_2015 = eval('DATASET_DIR_2015')
DATASET_DIR_2019 = eval('DATASET_DIR_2019')
INP_CONFIG_PATH = eval('INP_CONFIG_PATH')
MODEL_PATH = eval('MODEL_PATH')
NUM_EPOCHS = 25
BATCH_SIZE = 32
# auto-generated or doesn't need to change:
NUM_CLASSES = 5
OUT_CONFIG_PATH = 'config.json'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

with open(INP_CONFIG_PATH, 'r') as f:
    config = json.load(f)

IMAGE_SHAPE = config['preprocess15']['image_shape']

# ______________________________________________________________________________
# Classes & Functions:


# with CustomObjectScope({'kappa': kappa}):
class Metrics(Callback):

    def __init__(self, gen: MyGenerator, model_save_path='./model.h5', weights_save_path='./weights.h5'):
        super().__init__()
        self.gen: MyGenerator = gen
        self.y_true = multi_2_label(self.gen.labels)
        #
        self.model_save_path = model_save_path
        self.weights_save_path = weights_save_path

        self.kappa_history = []
        self.loss_history = []
        self.val_loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        # x_val_tmp, y_val_tmp = self.validation_data[:2]

        y_pred_evidence = self.model.predict_generator(self.gen)
        y_pred = multi_2_label(y_pred_evidence)

        _val_kappa = cohen_kappa_score(self.y_true, y_pred, weights='quadratic')
        print(f"val_kappa: {_val_kappa:.4f}".rjust(89))

        def _need_save(cur_kappa, logs, threshold=0.5):
            relation: np.array = logs['val_loss'] < np.array(self.val_loss_history)
            return relation.all()

        if _need_save(_val_kappa, logs, threshold=np.inf):
            print(">>>>>>>>>>>>>> Model has improved. Saving model.".rjust(89))
            self.model.save(self.model_save_path)
            self.model.save_weights(self.weights_save_path)
            # save_keras_model(self.model, './model_co.h5', custom_objects={'relu_max5': relu_max5})

        self.kappa_history.append(_val_kappa)
        self.loss_history.append(logs['loss'])
        self.val_loss_history.append(logs['val_loss'])

        return


kappa_opt = KappaOptimizer()


def build_model(input_shape, summary=True) -> keras.Model:
    """Create keras Model and Compile."""
    efn = EfficientNetB3(weights=None,
                         include_top=False,
                         input_shape=input_shape)
    efn.load_weights(MODEL_PATH)

    inp = layers.Input(shape=input_shape)
    x = efn(inp)
    x = layers.GlobalAveragePooling2D(name='last-pooling')(x)
    x = layers.Dropout(0.5, name='last-dropout')(x)
    out = layers.Dense(NUM_CLASSES, activation='sigmoid', name='out')(x)
    model = models.Model(inp, out, name='aptos-densenet')

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=[]
    )
    if summary:
        model.summary()

    return model


# ______________________________________________________________________________
# Preparations:
# #### read train.csv, test.csv ----> train_df, test_df
train_df_2015 = pd.read_csv(path.join(DATASET_DIR_2015, 'train.csv'))
train_df_2019 = pd.read_csv(path.join(DATASET_DIR_2019, 'train.csv'))
# #### Display sample images.
print("="*20 + " samples of dataset 2015")
rows, cols = 3, 4
sample_df = train_df_2015.sample(rows*cols)
show_images_given_paths(paths=pathj([DATASET_DIR_2015, 'train_images'], sample_df['id_code'], '.png'),
                        titles=sample_df['diagnosis'])
print("="*20 + " samples of dataset 2019")
sample_df = train_df_2019.sample(rows*cols)
show_images_given_paths(paths=pathj([DATASET_DIR_2019, 'train_images'], sample_df['id_code'], '.png'),
                        titles=sample_df['diagnosis'])

# REG_TABLE = np.array([0, 1, 2, 3, 4]) + 0.5
# INIT_KAPPA_COEF = list((REG_TABLE[:-1] + REG_TABLE[1:]) / 2)


# ______________________________________________________________________________
# Augmentations:
# #### augmentations:
def global_seq(): return iaa.Sequential(
    [
        iaa.Noop(),
    ],
    random_order=False
)


# noinspection PyTypeChecker
def aug_seq() : return iaa.Sequential(
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

# In[ ]:
# ## Train data or generator:
id_train, y_train = train_df_2015['id_code'], train_df_2015['diagnosis']
y_train = label_2_multi(y_train)
train_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR_2015, 'train_images'], id_train, '.png'),
    labels=y_train,
    batch_size=BATCH_SIZE,
    seqs=[global_seq(), aug_seq(), ],
    is_shuffle=True,
)
# ## Valid data or generator:
id_valid, y_valid = train_df_2019['id_code'], train_df_2019['diagnosis']
y_valid = label_2_multi(y_valid)
valid_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR_2019, 'train_images'], id_valid, '.png'),
    labels=y_valid,
    batch_size=BATCH_SIZE,
    seqs=[global_seq(), ],
    is_shuffle=False,
)
# x_valid = np.concatenate(list(map(lambda u: u[0], valid_gen)))

model = build_model(IMAGE_SHAPE)

kappa_metrics = Metrics(gen=valid_gen)
tbCallBack = TensorBoard(log_dir="./tensorboard_log", histogram_freq=4)

tic = datetime.now()
history = model.fit_generator(
    train_gen,
    epochs=NUM_EPOCHS,
    # Tensor Board requires: If printing histograms, validation_data must be provided, and cannot be a generator.
    validation_data=valid_gen,
    # validation_steps=ceil(x_valid.shape[0] / BATCH_SIZE),
    callbacks=[
        # Interrupt training if `val_loss` stops improving for over $patience epochs
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, mode='min'),
        # Track kappa,
        kappa_metrics,
        # Tensor Board log.
        # tbCallBack,
    ]
)
toc = datetime.now(); print("model.fit_generator spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))


# In[ ]:

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.figure()
plt.title('Kappa')
plt.plot(kappa_metrics.kappa_history)

# ______________________________________________________________________________
# fit kappa and post training.
print("loading model...")
# with custom_object_scope({'crossentropy_with_2logits': crossentropy_with_2logits}):
model: keras.Model = keras.models.load_model(
    'model.h5',
    custom_objects={'relu_max5': relu_max5}
)
model.summary()

tic = datetime.now()
y_pred_multi_valid = model.predict_generator(valid_gen)
toc = datetime.now(); print("model.predict spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))
y_pred_cate_valid = multi_2_label(y_pred_multi_valid)

# dump config for later use.
with open(OUT_CONFIG_PATH, 'w') as f:
    config['train15'] = {
        'build_model_code': inspect.getsource(build_model),
        # 'reg_table': list(REG_TABLE),
        # 'init_kappa_coef': INIT_KAPPA_COEF,
        # 'kappa_coef': kappa_opt.coef,
        'create_time': str(datetime.now(tz=REMOTE_TZ)),
    }
    json.dump(config, f)

# # copy preprocessing algorithm source code for later use.
# with tarfile.open(PREPROCESSED_TAR_FILE, 'r') as tar:
#     tar.extract(f'{ARCNAME}/prep_algo.py')




