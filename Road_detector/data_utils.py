import os
import sys
from os import path, listdir, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
from sklearn.model_selection import KFold
import cv2
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.callbacks import ModelCheckpoint
from resnet_unet import get_resnet_unet
from loss import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss
import skimage.io
import keras.backend as K

#from other part of data utils improt all_files, all_pan_files, all_masks, city_id

channel_no = 3
input_shape = (320, 320)
channel_no = 3
cities = ['Vegas','Paris']
city_datasets = dict(Vegas = 'AOI_2_Vegas_Roads_Train',
                     Paris = 'AOI_3_Paris_Roads_Train')
city_id = -1

# means = [[290.42, 446.84, 591.88], [178.33, 260.14, 287.4]]
# stds = [[75.42, 177.98, 288.81], [16.4, 45.69, 79.42]]

## define means and stds from reading data with npz format
def means_data(data):
    axis = tuple([i for i in data.shape[-1]])
    means = np.mean(data, axis = axis)
    print(means)
    return means

def stds_data(data):
    axis = tuple([i for i in data.shape[-1]])
    stds = np.std(data, axis = axis)
    print(stds)
    return stds

def preprocess_inputs_std(x, city_id):
    means = means_data(x)
    stds = stds_data(x)
    zero_msk = (x == 0)
    x = np.asarray(x, dtype='float32')
    for i in range(channel_no):
        x[..., i] -= means[city_id][i]
        x[..., i] /= stds[city_id][i]
    x[zero_msk] = 0
    return x

def datafiles(cities, city_datasets):
    all_files = []
    all_pan_files = []
    all_city_inp = []
    all_masks = []

    t0 = timeit.default_timer()

    fold_nums = [0, 1]

    # train_folders = []
    # for i in range(1, len(sys.argv)):
    #     train_folders.append(sys.argv[i])
    for city,d in city_datasets.items():
        masks_folder = os.path.join(os.getcwd(),'wdata/{}/masks_smallest'.format(d))
        models_folder = os.path.join(os.getcwd(),'wdata/{}/nn_models'.format(d))
        if not path.isdir(models_folder):
            mkdir(models_folder)
        for f in sorted(listdir(path.join(os.getcwd(), 'wdata', d, 'MUL'))):
            if path.isfile(path.join(os.getcwd(), 'wdata', d, 'MUL', f)) and '.tif' in f:
                img_id = f.split('MUL_')[1].split('.')[0]
                all_files.append(path.join(os.getcwd(), 'wdata', d, 'MUL', f))
                all_pan_files.append(path.join(os.getcwd(), 'wdata', d, 'PAN', 'PAN_{0}.tif'.format(img_id)))
                cinp = np.zeros((4,))
                cid = cities.index(img_id.split('_')[2])
                cinp[cid] = 1.0
                all_city_inp.append(cinp)
                all_masks.append(path.join(masks_folder, '{0}{1}'.format(img_id, '.png')))
    print(all_files[:2], all_pan_files[:2], all_city_inp[:2], all_masks[:2])
    all_files = np.asarray(all_files)
    all_pan_files = np.asarray(all_pan_files)
    all_city_inp = np.asarray(all_city_inp)
    all_masks = np.asarray(all_masks)
    return all_files, all_pan_files, all_city_inp, all_masks

# cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

def rotate_image(image, angle, scale):
    all_files, all_pan_files, all_city_inp, all_masks = datafiles(cities, city_datasets)
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result


def batch_data_generator(train_idx, batch_size):
    all_files, all_pan_files, all_city_inp, all_masks = datafiles(cities, city_datasets)
    inputs = []
    outputs = []
    while True:
        np.random.shuffle(train_idx)
        for i in train_idx:
            for j in range(1):
                img = skimage.io.imread(all_files[i], plugin='tifffile')
                pan = skimage.io.imread(all_pan_files[i], plugin='tifffile')
                pan = cv2.resize(pan, (325, 325))
                pan = pan[..., np.newaxis]
                # img = np.concatenate([img, pan], axis=2)
                rgb_index = [i for i in range(channel_no)]
                img = img[:, :, rgb_index]
                msk = cv2.imread(all_masks[i], cv2.IMREAD_UNCHANGED)[..., 0]

                if random.random() > 0.5:
                    scale = 0.9 + random.random() * 0.2
                    angle = random.randint(0, 41) - 24
                    img = rotate_image(img, angle, scale)
                    msk = rotate_image(msk, angle, scale)

                x0 = random.randint(0, img.shape[1] - input_shape[1])
                y0 = random.randint(0, img.shape[0] - input_shape[0])
                img = img[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                msk = (msk > 127) * 1
                msk = msk[..., np.newaxis]
                otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1], :]

                if random.random() > 0.5:
                    img = img[:, ::-1, ...]
                    otp = otp[:, ::-1, ...]

                rot = random.randrange(4)
                if rot > 0:
                    img = np.rot90(img, k=rot)
                    otp = np.rot90(otp, k=rot)

                inputs.append(img)
                outputs.append(otp)

                if len(inputs) == batch_size:
                    inputs = np.asarray(inputs)
                    outputs = np.asarray(outputs, dtype='float')
                    inputs = preprocess_inputs_std(inputs, city_id)
                    yield inputs, outputs
                    inputs = []
                    outputs = []

def val_data_generator(val_idx, batch_size, validation_steps):
    all_files, all_pan_files, all_city_inp, all_masks = datafiles(cities, city_datasets)
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for i in val_idx:
            img0 = skimage.io.imread(all_files[i], plugin='tifffile')
            pan = skimage.io.imread(all_pan_files[i], plugin='tifffile')
            pan = cv2.resize(pan, (325, 325))
            pan = pan[..., np.newaxis]
            # img0 = np.concatenate([img0, pan], axis=2)
            rgb_index = [i for i in range(channel_no)]
            img0 = img0[:, :, rgb_index]
            msk = cv2.imread(all_masks[i], cv2.IMREAD_UNCHANGED)[..., 0:1]
            msk = (msk > 127) * 1
            for x0, y0 in [(0, 0)]:
                img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                inputs.append(img)
                outputs.append(otp)
                if len(inputs) == batch_size:
                    step_id += 1
                    inputs = np.asarray(inputs)
                    outputs = np.asarray(outputs, dtype='float')
                    inputs = preprocess_inputs_std(inputs, city_id)
                    yield inputs, outputs
                    inputs = []
                    outputs = []
                    if step_id == validation_steps:
                        break
