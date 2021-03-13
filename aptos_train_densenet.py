# coding: utf-8

# In[ ]:
import inspect
import json
from datetime import datetime
from os import path

import pandas as pd
import pytz
from imgaug.parameters import *
from keras.applications import *
from keras.callbacks import Callback
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from aptos_utils import *

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:
# ______________________________________________________________________________
# Config:
# #### set by user:
MODEL_PATH = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'
NUM_EPOCHS = 50
BATCH_SIZE = 32
# #### auto-generated or doesn't need to change:
NUM_CLASSES = 5
DATASET_DIR = eval('DATASET_DIR')
INP_CONFIG_PATH = path.join(eval('PREP_DIR'), 'config.json')
OUT_CONFIG_PATH = './config.json'
REMOTE_TZ = pytz.timezone('Asia/Shanghai')

with open(INP_CONFIG_PATH, 'r') as f:
    config = json.load(f)

IMAGE_SHAPE = config['preprocess']['image_shape']

# In[ ]:

# ## read train.csv, test.csv ----> train_df, test_df
train_valid_df = pd.read_csv(path.join(DATASET_DIR, 'train.csv'))
test_df = pd.read_csv(path.join(DATASET_DIR, 'test.csv'))
# ## Display sample images.
rows, cols = 3, 4
sample_df = train_valid_df.sample(rows * cols)
show_images_given_paths(paths=pathj([DATASET_DIR, 'train_images'], sample_df['id_code'], '.png'),
                        titles=sample_df['diagnosis'])

# In[ ]:
# Now: diagnosis_list has shape: [n_samples, NUM_CLASSES]
diagnosis_list = np.array([eval(x) for x in train_valid_df['diagnosis']])
# Now: y_train_valid has shape: [n_samples, ]

# REG_TABLE = (np.array([0, 1, 2, 3, 4]) + 0.5) / 5
REG_TABLE = np.array([0, 1, 2, 3, 4])
INIT_KAPPA_COEF = list((REG_TABLE[:-1] + REG_TABLE[1:]) / 2)

y_train_valid = label_counts_2_regression(diagnosis_list, regression_table=REG_TABLE)
y_true_category = KappaOptimizer.regression_2_category(y_train_valid, coef=INIT_KAPPA_COEF)

id_train, id_valid, y_train, y_valid = train_test_split(
    np.array(train_valid_df['id_code']), y_train_valid,
    test_size=0.15,
    # stratify=y_true_category,
    random_state=2019
)

x_train = np.array(list(tqdm(map(
    lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
    pathj([DATASET_DIR, 'train_images'], id_train, extname='.png')
))))
x_valid = np.array(list(tqdm(map(
    lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
    pathj([DATASET_DIR, 'train_images'], id_valid, extname='.png')
))))


# In[ ]:


def aug_seq():
    return iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # crop images by -5% to 10% of their height/width
            iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                # rotate=Clip(Normal(0, 5), minval=-10, maxval=10),  # rotate by -45 to +45 degrees
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
            iaa.Multiply(Clip(Normal(1, 0.1), minval=0.8, maxval=1))
        ],
        random_order=False
    )




train_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'train_images'], id_train, '.png'),
    labels=y_train,
    batch_size=BATCH_SIZE,
    seqs=[aug_seq(), ],
    # rescale=1.0 / 255,
    is_shuffle=True,
)
valid_gen = MyGenerator(
    img_paths=pathj([DATASET_DIR, 'train_images'], id_valid, '.png'),
    labels=y_valid,
    batch_size=BATCH_SIZE,
    # rescale=1.0 / 255,
    is_shuffle=False,
)

kappa_opt = KappaOptimizer()


# In[ ]:

# with CustomObjectScope({'kappa': kappa}):
class Metrics(Callback):

    def __init__(self):
        super().__init__()
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
        y_pred = KappaOptimizer.regression_2_category(y_pred_evidence, coef=INIT_KAPPA_COEF)
        y_true = KappaOptimizer.regression_2_category(y_val_tmp, coef=INIT_KAPPA_COEF)

        _val_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        print(f"val_kappa: {_val_kappa:.4f}".rjust(89))

        def _need_save(cur_kappa, logs, threshold=0.5):
            relation: np.array = logs['val_loss'] < np.array(self.val_loss_history)
            return relation.all()

        if _need_save(_val_kappa, logs, threshold=np.inf):
            print(">>>>>>>>>>>>>> Model has improved. Saving model.".rjust(89))
            self.model.save('model.h5')

        self.kappa_history.append(_val_kappa)
        self.loss_history.append(logs['loss'])
        self.val_loss_history.append(logs['val_loss'])

        return


def build_model(input_shape) -> keras.Model:
    inp = layers.Input(shape=input_shape)
    x = DenseNet121(weights=MODEL_PATH,
                    include_top=False,
                    input_shape=input_shape)(inp)
    x = layers.GlobalAveragePooling2D(name='last-pooling')(x)
    x = layers.BatchNormalization(name='last-bn')(x)
    x = layers.Dense(1024, activation='relu', name='fc')(x)
    x = layers.Dropout(0.5, name='last-dropout')(x)
    out = layers.Dense(1, activation='relu', name='out')(x)
    model = models.Model(inp, out, name='aptos-densenet')

    return model


model = build_model(IMAGE_SHAPE)
# ______________________________________________________________________________
# #### train all layers
# for layer in model.layers:
#     layer.trainable = True

model.compile(loss='mse',
              optimizer=Adam(lr=5e-5),
              metrics=[])
model.summary()

kappa_metrics = Metrics()

tic = datetime.now()
history = model.fit_generator(
    train_gen,
    # steps_per_epoch=len(train_data_iterator),
    epochs=NUM_EPOCHS,
    # Tensor Board requires: If printing histograms, validation_data must be provided, and cannot be a generator.
    validation_data=(x_valid, y_valid),
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
model_best: keras.Model = keras.models.load_model(
    'model.h5',
    custom_objects={'relu_max5': relu_max5},
)
model_best.summary()

y_pred_regression = model_best.predict(x_valid, verbose=1)
y_valid_true = KappaOptimizer.regression_2_category(y_valid, coef=INIT_KAPPA_COEF)

kappa_opt.fit(y_pred_regression, y_valid_true)

# dump config for later use.
with open(OUT_CONFIG_PATH, 'w') as f:
    config['train'] = {
        'build_model_code': inspect.getsource(build_model),
        'init_kappa_coef': INIT_KAPPA_COEF,
        'kappa_coef': kappa_opt.coef,
        'create_time': str(datetime.now(tz=REMOTE_TZ)),
    }
    json.dump(config, f)

# # copy preprocessing algorithm source code for later use.
# with tarfile.open(PREPROCESSED_TAR_FILE, 'r') as tar:
#     tar.extract(f'{ARCNAME}/prep_algo.py')
