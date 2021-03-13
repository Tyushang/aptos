#
# from datetime import datetime
#
# import json
# import pytz
# import pandas as pd
#
# try:
#     sub = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
# except:
#     sub = pd.read_csv('../input/test.csv')
#
# if len(sub) < 2000:
#     # Make sure MODEL_NAME_2 equals MODEL_NAME manually!
#     MODEL_NAME_2 = 'aptos-train-densenet'
#     REMOTE_TZ = pytz.timezone('Asia/Shanghai')
#     # write config.json
#     with open(f'../input/{MODEL_NAME_2}/config.json') as f:
#         config = json.load(f)
#     with open('config.json', 'w') as f:
#         config['inference'] = {
#             'create_time': str(datetime.now(tz=REMOTE_TZ)),
#         }
#         json.dump(config, f)
#     print("==== dummy commit ...")
#     sub.to_csv('submission.csv' ,index=False)
#     exit()



# In[ ]:
# ______________________________________________________________________________
# For train: extract all files in preprocessed tar.

# from os import path
# import tarfile
#
# PREP_DIR = '../input/aptos-preprocess-crop'
# # auto-generated or doesn't need to change:
# ARCNAME = 'aptos-preprocessed'
# PREP_TAR_FILE = path.join(PREP_DIR, ARCNAME + '.tar')
# DATASET_EXTRACT_DIR = '/tmp'
# # For cells below use:
# DATASET_DIR = path.join(DATASET_EXTRACT_DIR, ARCNAME)
# INP_CONFIG_PATH = path.join(PREP_DIR, 'config.json')
#
# with tarfile.open(PREP_TAR_FILE, 'r') as tar:
#     tar.extractall(DATASET_EXTRACT_DIR)
#
# !ls $DATASET_DIR


# In[ ]:
# ______________________________________________________________________________
# For preprocess old: deduplication
#
# def img_id_to_path(id: str, img_dir:str=IMAGE_2015_DIR):
#     return path.join(img_dir, id + '.jpeg')
#
#
# def get_md5_digest(file_path):
#     with open(file_path, 'rb') as f:
#         m = hashlib.md5()
#         while True:
#             str_read = f.read(0x1000)
#             if len(str_read) == 0:
#                 break
#             m.update(str_read)
#         return m.hexdigest()
#
#
# # Calc MD5.
# with Pool() as p:
#     md5_digest_list = list(tqdm(p.imap(
#         get_md5_digest,
#         [img_id_to_path(id) for id in train_df['id_code']],
#         chunksize = 100,
#     )))
# train_df['md5'] = md5_digest_list
# print(train_df)
#
# gb = train_df.groupby('md5')
# #
# dup_dict = {}
# for key, group in gb:
#     dup_degree = len(group)
#     if dup_dict.get(dup_degree) is None:
#         dup_dict[dup_degree] = []
#     dup_dict[dup_degree].append(group)
# # duplication summary.
# for deg, group_list in dup_dict.items():
#     print(f'dup degree: {deg}; num groups: {len(group_list)}')
#
#
# def plot_dup(df: pd.DataFrame):
#     print('==== i = a = m = s = p = l = i = t = e = r ===='*3)
#     cols = 2
#     rows = ceil(len(df) / cols)
#     fig = plt.figure(figsize = (8*cols, 6*rows))
#     for i, t in enumerate(df.itertuples()):
#         img_path = img_id_to_path(t.id_code)
#         sp = fig.add_subplot(rows, cols, i + 1)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         sp.set_title(t.id_code + f' Label: {t.diagnosis}')
#         sp.imshow(img)
#     plt.show()
#
#
# # visual duplicated images.
# for df in dup_dict[2]:
#     plot_dup(df)
# for df in dup_dict[3]:
#     plot_dup(df)
# for df in dup_dict[4]:
#     plot_dup(df)
#
# # deduplication algorithm.
# frames = []
# for key, group in gb:
#     df = pd.DataFrame()
#     df['id_code'] = group.head(1)['id_code']
#     df['diagnosis'] = group['diagnosis'].mean()
#     frames.append(df)
# dedup_train_df = pd.concat(frames).sort_index()


# def map_func(img_array):
#     # img = Image.open(path.join(DATASET_DIR, 'train_images', img_id + '.png'))
#     arr = img_array.astype('float64') # np.asarray(img_array, dtype='float64')
#     mean, std = arr.mean(), arr.std()
#     return ((arr - mean) / std).astype('float32')


# def label_counts_2_logits(label_counts, multiplier=1.0, bias=0.0):
#     logits_of_NoDR = np.array([10,9,7,4,0])
#     cir = circulant(logits_of_NoDR)
#     logits_table = np.tril(cir) + np.tril(cir, k=-1).T
#     # label_counts' shape: [n_samples, NUM_CLASSES]
#     label_counts = np.array(label_counts, dtype=int)
#     # return value shape: [n_samples, NUM_CLASSES]
#     return multiplier * np.matmul(label_counts, logits_table) + bias


# def preprocessing_algorithm(img_path, desired_size=224, resample=Image.LANCZOS):
#     img = Image.open(img_path)
#     return np.array(img.resize((desired_size,) * 2, resample=resample))
# handler = preprocessing_algorithm


# In[]:
# import numpy as np
# import imgaug as ia
# import imgaug.augmenters as iaa
#
#
# # random example images
# images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
#
# # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#
# # Define our sequence of augmentation steps that will be applied to every image
# # All augmenters with per_channel=0.5 will sample one value _per image_
# # in 50% of all cases. In all other cases they will sample new values
# # _per channel_.
# seq = iaa.Sequential(
#     [
#         # apply the following augmenters to most images
#         iaa.Fliplr(0.5), # horizontally flip 50% of all images
#         iaa.Flipud(0.2), # vertically flip 20% of all images
#         # crop images by -5% to 10% of their height/width
#         sometimes(iaa.CropAndPad(
#             percent=(-0.05, 0.1),
#             pad_mode=ia.ALL,
#             pad_cval=(0, 255)
#         )),
#         sometimes(iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
#             rotate=(-45, 45), # rotate by -45 to +45 degrees
#             shear=(-16, 16), # shear by -16 to +16 degrees
#             order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#             cval=(0, 255), # if mode is constant, use a cval between 0 and 255
#             mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         )),
#         # execute 0 to 5 of the following (less important) augmenters per image
#         # don't execute all of them, as that would often be way too strong
#         iaa.SomeOf((0, 5),
#             [
#                 sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
#                 iaa.OneOf([
#                     iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
#                     iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
#                     iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
#                 ]),
#                 iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                 iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
#                 # search either for all edges or for directed edges,
#                 # blend the result with the original image using a blobby mask
#                 iaa.SimplexNoiseAlpha(iaa.OneOf([
#                     iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
#                 ])),
#                 iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
#                 iaa.OneOf([
#                     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#                     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
#                 ]),
#                 iaa.Invert(0.05, per_channel=True), # invert color channels
#                 iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
#                 iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
#                 # either change the brightness of the whole image (sometimes
#                 # per channel) or change the brightness of subareas
#                 iaa.OneOf([
#                     iaa.Multiply((0.5, 1.5), per_channel=0.5),
#                     iaa.FrequencyNoiseAlpha(
#                         exponent=(-4, 0),
#                         first=iaa.Multiply((0.5, 1.5), per_channel=True),
#                         second=iaa.ContrastNormalization((0.5, 2.0))
#                     )
#                 ]),
#                 iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
#                 iaa.Grayscale(alpha=(0.0, 1.0)),
#                 sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
#                 sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
#                 sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
#             ],
#             random_order=True
#         )
#     ],
#     random_order=True
# )
#
# images_aug = seq.augment_images(images)


# In[]:
# # Two example plots
# from matplotlib.ticker import MultipleLocator
# fig = plt.figure(figsize=(20,10))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
# ref = (list(range(50)), list(range(50)), 'r--')
# # Set minor tick locations.
# spacing = 5 # This can be your user specified spacing.
# minorLocator = MultipleLocator(spacing)
#
# ax1.plot(tt, yt, 'g^', *ref )
# ax1.yaxis.set_minor_locator(minorLocator)
# ax1.xaxis.set_minor_locator(minorLocator)
# ax1.grid(which = 'minor')
#
# ax2.plot(vv, yv, 'bs', *ref)
# ax2.yaxis.set_minor_locator(minorLocator)
# ax2.xaxis.set_minor_locator(minorLocator)
# ax2.grid(which = 'minor')
#
# plt.tight_layout()



















