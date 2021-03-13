# coding: utf-8

# In[ ]:
import inspect
import json
from datetime import datetime
from os import path

import pytz
import tensorflow as tf
from efficientnet import EfficientNetB3
from imgaug.parameters import *
from keras.applications import *
from keras.callbacks import Callback
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from aptos_utils import *

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:
# ______________________________________________________________________________
# Config:
# #### set by user:
NUM_EPOCHS = 15
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
        # y_pred = np.argmax(y_pred_logits, axis=-1)
        # y_true = np.argmax(y_val_tmp, axis=-1)
        print(f"adaptive reg table: {get_adapt_reg_table()}")
        coef = get_adapt_coef()
        print(f"adaptive coef: {coef}")
        y_pred = KappaOptimizer.regression_2_category(y_pred_evidence, coef=coef)
        y_true = y_val_tmp # KappaOptimizer.regression_2_category(y_val_tmp, coef=coef)

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
        x = layers.BatchNormalization(name='last-bn')(x)
        x = layers.Dense(1024, activation='relu', name='fc')(x)
        x = layers.Dropout(0.5, name='last-dropout')(x)
        out = layers.Dense(1, activation=relu_max5, name='out')(x)
        model = models.Model(inp, out, name='aptos-densenet')

        model.compile(
            loss=adapt_mse,
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
train_valid_df = pd.read_csv(path.join(DATASET_DIR, 'train.csv')).sample(500)
test_df = pd.read_csv(path.join(DATASET_DIR, 'test.csv'))
# #### Display sample images.
rows, cols = 3, 4
sample_df = train_valid_df.sample(rows*cols)
# show_images_given_paths(paths=pathj([DATASET_DIR, 'train_images'], sample_df['id_code'], '.png'),
#                         titles=sample_df['diagnosis'])

# REG_TABLE = (np.array([0, 1, 2, 3, 4]) + 0.5) / 5
REG_TABLE = np.array([0, 1, 2, 3, 4]) + 0.5
INIT_KAPPA_COEF = list((REG_TABLE[:-1] + REG_TABLE[1:]) / 2)

y_true_cate_train_valid = np.array(train_valid_df['diagnosis'])
y_true_reg_train_valid = REG_TABLE[y_true_cate_train_valid]

pred_buff = [
    [REG_TABLE[0]]*32,
    [REG_TABLE[1]]*32,
    [REG_TABLE[2]]*32,
    [REG_TABLE[3]]*32,
    [REG_TABLE[4]]*32,
]


def get_adapt_reg_table() -> np.ndarray:
    return np.array(pred_buff).mean(axis=-1)


def get_adapt_coef() -> np.ndarray:
    table = get_adapt_reg_table()
    return (table[:-1] + table[1:]) / 2


def get_adapt_mean(y_true: List, y_pred: List) -> np.ndarray:
    # update pred_buffer.
    for cate, pred in zip(y_true, y_pred):
        del pred_buff[cate][0]
        pred_buff[cate].append(pred)
    # get adapt reg table.
    table = get_adapt_reg_table()

    return table[y_true]

from tensorflow import contrib
autograph = contrib.autograph

@autograph.convert(optional_features=autograph.Feature.ALL)
def adapt_mse(y_true, y_pred):
    loss = 0.0
    # autograph.set_element_type(loss, tf.float32)
    # autograph.set_element_type(y_pred, tf.float32)
    # for t, p in zip(y_true, y_pred):
    #     print(f"true: {t}; pred: {p}")
    #     loss += (p - t)**2
    for p in y_pred:
        loss = (p)**2
    return loss


# def adapt_mse(y_true: tf.Tensor, y_pred: tf.Tensor):
#     from keras import backend as K
#
#     def cond(i):
#
#
#     for i in range(int(y_shape[0])):
#         print(y_true[i])
#
#     y_true_eval = list(K.eval(y_true))
#     y_pred_eval = list(K.eval(y_pred))
#     adapt_mean = get_adapt_mean(y_true_eval, y_pred_eval)
#     return K.mean(K.square(y_pred - y_true), axis=-1)


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

class AdaptiveGenerator(MyGenerator):
    CATE = [0, 1, 2, 3, 4]

    def __init__(self,
                 img_paths: List[str],
                 labels: List[float] = None,
                 coef=(1, 2, 3, 4),
                 batch_size: int = 32,
                 y_true_cate_valid: List[int] = None,
                 prep_algo: Callable[[np.ndarray], np.ndarray] = None,
                 prep_kw: dict = None,
                 seqs: List[iaa.Sequential] = None,
                 rescale=1.0,
                 is_shuffle=False):
        super().__init__(2*list(img_paths), batch_size, 2*list(labels), prep_algo, prep_kw,
                         seqs, rescale, is_shuffle, is_mixup=False)
        # copy origin img_paths and labels.
        self.origin_img_paths = img_paths.copy()
        self.origin_y_true_reg_train = labels.copy()
        self.origin_y_true_cate_train = \
            KappaOptimizer.regression_2_category(self.origin_y_true_reg_train, coef=coef)
        self.origin_num_samples = len(self.origin_img_paths)
        self.group_by_cate = pd.DataFrame({'img': self.origin_img_paths,
                                           'label': self.origin_y_true_reg_train,
                                           'cate': self.origin_y_true_cate_train}).groupby('cate')
        # determine next epoch is adaptive or origin.
        self.is_next_epoch_adaptive = True
        # queue to receive y_valid_pred
        self.queue = queue.Queue()
        self.y_true_cate_valid = np.array(y_true_cate_valid).flatten()

    def on_epoch_end(self):
        y_pred_cate_valid = np.array(self.queue.get()).flatten()
        # origin epoch and adaptive epoch runs alternately.
        # if self.is_next_epoch_adaptive:
        #     self.img_paths, self.labels = self._resample(y_pred_cate_valid)
        # else:
        #     self.img_paths, self.labels = self.origin_img_paths, self.origin_y_true_reg_train
        adapt_img_paths, adapt_y_true_reg_train = self._resample(y_pred_cate_valid)
        self.img_paths = self.origin_img_paths + adapt_img_paths
        self.labels = self.origin_y_true_reg_train + adapt_y_true_reg_train
        # shuffle if required.
        if self.is_shuffle:
            self._shuffle_paths_labels()
        # self.is_next_epoch_adaptive = not self.is_next_epoch_adaptive

        self._visual(y_pred_cate_valid)
        self.queue.task_done()

    def _resample(self, y_pred_cate_valid) -> Tuple[List[str], List[float]]:
        """resample images by prob."""
        # resample probability for every label.
        resample_prob = self._get_resample_prob(y_pred_cate_valid)
        # resample times for every label.
        label_freq = self._prob_2_freq(resample_prob)
        # do resample according to label_freq.
        frames = []
        for cate, group_df in self.group_by_cate:
            frames.append(group_df.sample(label_freq[cate], replace=True))
        resampled_df: pd.DataFrame = shuffle(pd.concat(frames))

        return list(resampled_df['img']), list(resampled_df['label'])

    def _get_resample_prob(self, y_pred_cate_valid):
        """if predict is right, nothing to do; else, contribute to resample prob of both label(pred and true)."""
        idx = self.y_true_cate_valid != y_pred_cate_valid
        cat = np.concatenate((self.y_true_cate_valid[idx], y_pred_cate_valid[idx]))
        counts = np.bincount(cat)
        return counts / counts.sum()

    def _prob_2_freq(self, prob):
        """same with prob*self.num_samples theoretically."""
        label_samples = np.random.choice(self.CATE, self.origin_num_samples, p=prob)
        return np.bincount(label_samples)

    def _visual(self, y_pred_cate_valid):
        print(confusion_matrix(self.y_true_cate_valid, y_pred_cate_valid))
        df = pd.DataFrame({'label': self.labels})
        print(df['label'].value_counts().sort_index())
        df['label'].hist()


# ______________________________________________________________________________
# #### K-Fold CV:
kf = GroupKFold(n_splits=5, ) # shuffle=True, random_state=2019)

ids = np.array(train_valid_df['id_code'])
md5 = np.array(train_valid_df['md5'])

for i_fold, (train_index, valid_index) in enumerate(kf.split(ids, y_true_cate_train_valid, groups=md5)):
    # y_true_cate_valid = y_true_cate_train_valid[valid_index]
    # ## Train data or generator:
    id_train, y_true_cate_train = ids[train_index], y_true_cate_train_valid[train_index]
    train_gen = MyGenerator(
        img_paths=pathj([DATASET_DIR, 'train_images'], id_train, '.png'),
        labels=list(y_true_cate_train),
        # coef=INIT_KAPPA_COEF,
        # y_true_cate_valid=list(y_true_cate_valid),
        batch_size=BATCH_SIZE,
        seqs=[global_seq(), aug_seq(), ],
        # rescale=1.0 / 255,
        is_shuffle=True,
    )
    # ## Valid data or generator:
    id_valid, y_true_cate_valid = ids[valid_index], y_true_cate_train_valid[valid_index]
    valid_gen = MyGenerator(
        img_paths=pathj([DATASET_DIR, 'train_images'], id_valid, '.png'),
        labels=list(y_true_cate_valid),
        batch_size=BATCH_SIZE,
        seqs=[global_seq(), ],
        # rescale=1.0 / 255,
        is_shuffle=False,
    )
    x_valid = np.concatenate(list(map(lambda u: u[0], valid_gen)))

    # ## Create model and training...
    print('==== i = a = m = s = p = l = i = t = e = r ====' * 2)
    try:
        model = eval('modelxxx')
    except:
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
        validation_data=(x_valid, y_true_cate_valid),
        callbacks=[
            # Interrupt training if `val_loss` stops improving for over $patience epochs
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min'),
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
y_true_cate_train = eval('y_true_reg_train')
y_true_cate_valid = eval('y_true_reg_valid')

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

y_pred_regression = model_best.predict(x_valid, verbose=1)
y_valid_true = KappaOptimizer.regression_2_category(y_true_cate_valid, coef=INIT_KAPPA_COEF)

kappa_opt.fit(y_pred_regression, y_valid_true)

# dump config for later use.
with open(OUT_CONFIG_PATH, 'w') as f:
    config['train'] = {
        'build_model_code': inspect.getsource(build_model),
        'init_kappa_coef': INIT_KAPPA_COEF,
        # 'kappa_coef': kappa_opt.coef,
        'create_time': str(datetime.now(tz=REMOTE_TZ)),
    }
    json.dump(config, f)



