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
from keras.applications import *
from keras.callbacks import Callback
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, GroupKFold

from aptos_utils import *

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:
# ______________________________________________________________________________
# Config:
# #### set by user:
NUM_EPOCHS = 30
BATCH_SIZE = 32
# #### auto-generated or doesn't need to change:
TRAINED_BY_OLD = eval('TRAINED_BY_OLD')
MODEL_PATH = eval('MODEL_PATH')
IS_CV = eval('IS_CV')
DATASET_DIR = eval('DATASET_DIR')
INP_CONFIG_PATH = path.join(eval('PREP_DIR'), 'config.json')

OUT_CONFIG_PATH = './config.json'
NUM_CLASSES = 5
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

with open(INP_CONFIG_PATH, 'r') as f:
    config = json.load(f)

IMAGE_SHAPE = config['preprocess']['image_shape']


# In[ ]:
# ______________________________________________________________________________
# Classes & Functions:

# with CustomObjectScope({'kappa': kappa}):
class Metrics(Callback):

    def __init__(self, adaptive_gen=None, model_save_path='./model.h5'):
        super().__init__()
        self.adaptive_gen = adaptive_gen
        self.model_save_path = model_save_path
        # validation data iterator
        #
        self.kappa_history = []
        self.loss_history = []
        self.val_loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        x_val_tmp, y_val_tmp = self.validation_data[:2]

        y_pred_evidence = self.model.predict(x_val_tmp)
        y_pred = multi_2_label(y_pred_evidence)
        y_true = multi_2_label(y_val_tmp)

        if self.adaptive_gen is not None:
            self._send_preds(y_pred)

        _val_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        print(f"val_kappa: {_val_kappa:.4f}".rjust(89))

        def _need_save(cur_kappa, logs, threshold=0.5):
            relation: np.array = logs['val_loss'] < np.array(self.val_loss_history)
            return relation.all()

        if _need_save(_val_kappa, logs, threshold=np.inf):
            print(">>>>>>>>>>>>>> Model has improved. Saving model.".rjust(89))
            self.model.save(self.model_save_path)

        self.kappa_history.append(_val_kappa)
        self.loss_history.append(logs['loss'])
        self.val_loss_history.append(logs['val_loss'])

        return

    def _send_preds(self, y_true_cate_valid):
        self.adaptive_gen.queue.put(y_true_cate_valid)
        self.adaptive_gen.queue.join()


kappa_opt = KappaOptimizer()


def build_model(trained_model_path=MODEL_PATH,
                trained_model_type='weights',
                input_shape=IMAGE_SHAPE,
                custom_objects:dict=None,
                summary=True) -> keras.Model:
    """Create keras Model and Compile."""
    if trained_model_type == 'weights':
        efn = EfficientNetB3(weights=None,
                             include_top=False,
                             input_shape=input_shape)
        efn.load_weights(MODEL_PATH)

        inp = layers.Input(shape=input_shape)
        x = efn(inp)
        x = layers.GlobalAveragePooling2D(name='last-pooling')(x)
        # x = layers.BatchNormalization(name='last-bn')(x)
        # x = layers.Dense(1024, activation='relu', name='fc')(x)
        x = layers.Dropout(0.5, name='last-dropout')(x)
        out = layers.Dense(NUM_CLASSES, activation='sigmoid', name='out')(x)
        model = models.Model(inp, out, name='aptos-densenet')

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.00005),
            metrics=[]
        )
    else:
        model = keras.models.load_model(
            trained_model_path,
            custom_objects={'relu_max5': relu_max5},
        )

    if summary:
        model.summary()

    return model



# In[ ]:
# ______________________________________________________________________________
# Preparation:
# #### read train.csv, test.csv ----> train_df, test_df
train_valid_df = pd.read_csv(path.join(DATASET_DIR, 'train.csv'))
test_df = pd.read_csv(path.join(DATASET_DIR, 'test.csv'))
# #### Display sample images.
rows, cols = 3, 4
sample_df = train_valid_df.sample(rows*cols)
show_images_given_paths(paths=pathj([DATASET_DIR, 'train_images'], sample_df['id_code'], '.png'),
                        titles=sample_df['diagnosis'])

# REG_TABLE = (np.array([0, 1, 2, 3, 4]) + 0.5) / 5
# REG_TABLE = np.array([0, 1, 2, 3, 4]) + 0.5
# INIT_KAPPA_COEF = list((REG_TABLE[:-1] + REG_TABLE[1:]) / 2)

y_true_cate_train_valid = np.array(train_valid_df['diagnosis'])
y_true_multi_train_valid = label_2_multi(y_true_cate_train_valid)


# In[ ]:
# ______________________________________________________________________________
# Train:
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


# ______________________________________________________________________________
# #### K-Fold CV:
kf = GroupKFold(n_splits=5, ) # shuffle=True, random_state=2019)

ids = np.array(train_valid_df['id_code'])
md5 = np.array(train_valid_df['md5'])
for i_fold, (train_index, valid_index) in enumerate(kf.split(ids, y_true_multi_train_valid, groups=md5)):
    # ## Train data or generator:
    id_train, y_true_multi_train = ids[train_index], y_true_multi_train_valid[train_index]
    train_gen = MyGenerator(
        img_paths=pathj([DATASET_DIR, 'train_images'], id_train, '.png'),
        labels=y_true_multi_train,
        batch_size=BATCH_SIZE,
        seqs=[global_seq(), aug_seq(), ],
        # rescale=1.0 / 255,
        is_shuffle=True,
    )
    # ## Valid data or generator:
    id_valid, y_true_multi_valid = ids[valid_index], y_true_multi_train_valid[valid_index]
    valid_gen = MyGenerator(
        img_paths=pathj([DATASET_DIR, 'train_images'], id_valid, '.png'),
        labels=y_true_multi_valid,
        batch_size=BATCH_SIZE,
        seqs=[global_seq(), ],
        # rescale=1.0 / 255,
        is_shuffle=False,
    )
    x_valid = np.concatenate(list(map(lambda u: u[0], valid_gen)))

    # ## Create model and training...
    print('==== i = a = m = s = p = l = i = t = e = r ====' * 2)
    model = build_model(trained_model_path=MODEL_PATH,
                        trained_model_type='model' if TRAINED_BY_OLD else 'weights',
                        input_shape=IMAGE_SHAPE, summary=(i_fold == 0))
    kappa_metrics = Metrics(model_save_path=f'model_{i_fold}.h5')

    tic = datetime.now()
    history = model.fit_generator(
        train_gen,
        # steps_per_epoch=len(train_data_iterator),
        epochs=NUM_EPOCHS,
        # Tensor Board requires: If printing histograms, validation_data must be provided, and cannot be a generator.
        validation_data=(x_valid, y_true_multi_valid),
        callbacks=[
            # Interrupt training if `val_loss` stops improving for over $patience epochs
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, mode='min'),
            # Track kappa,
            kappa_metrics,
            # Tensor Board log.
            # TensorBoard(log_dir="./tensorboard_log", histogram_freq=4),
        ],
        verbose=1,
    )
    toc = datetime.now(); print("model.fit_generator spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))
    # ## Plot history and metrics:
    with open('history.json', 'w') as f:
        json.dump(history.history, f)

    history_df = pd.DataFrame(history.history)
    history_df[['loss', 'val_loss']].plot()
    plt.figure()
    plt.title('Kappa')
    plt.plot(kappa_metrics.kappa_history)

    if not IS_CV:
        break
# ______________________________________________________________________________
# For compatibility.
id_train = eval('id_train')
id_valid = eval('id_valid')
x_train = np.array(list(tqdm(map(
    lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
    pathj([DATASET_DIR, 'train_images'], id_train, extname='.png')
))))
x_valid = np.array(list(tqdm(map(
    lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
    pathj([DATASET_DIR, 'train_images'], id_valid, extname='.png')
))))
y_true_multi_train = eval('y_true_multi_train')
y_true_multi_valid = eval('y_true_multi_valid')

model_path = './model_0.h5'
# ______________________________________________________________________________
# fit kappa and post training.
print("loading model...")
# with custom_object_scope({'crossentropy_with_2logits': crossentropy_with_2logits}):
model_best: keras.Model = keras.models.load_model(
    model_path,
    custom_objects={'relu_max5': relu_max5},
)
model_best.summary()

y_pred_multi_valid = model_best.predict(x_valid, verbose=1)
y_pred_cate_valid = multi_2_label(y_pred_multi_valid)
print(y_pred_cate_valid)

# kappa_opt.fit(y_pred_regression, y_valid_true)

# dump config for later use.
with open(OUT_CONFIG_PATH, 'w') as f:
    config['train'] = {
        'build_model_code': inspect.getsource(build_model),
        # 'init_kappa_coef': INIT_KAPPA_COEF,
        # 'kappa_coef': kappa_opt.coef,
        'create_time': str(datetime.now(tz=REMOTE_TZ)),
    }
    json.dump(config, f)



