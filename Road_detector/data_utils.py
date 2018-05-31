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
from PIL import Image
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
origin_shape = (325, 325)
img_head = 'RGB-PanSharpen_'
# rgb_index = [5, 4, 3]
rgb_index = [0, 1, 2]

## define means and stds from reading data with npz format
def means_data(data):
    axis = tuple([i for i in range(data.shape[-1])])
    means = np.mean(data, axis = axis)
    # print(means)
    return means

def stds_data(data):
    axis = tuple([i for i in range(data.shape[-1])])
    stds = np.std(data, axis = axis)
    # print(stds)
    return stds

def color_scale(arr):
    """correct the wv-3 bands to be a composed bands of value between 0 255"""
    axis = tuple([i for i in range(arr.shape[-1])])
    str_arr = (arr - np.min(arr, axis = axis))*255.0/(np.max(arr, axis = axis) - np.min(arr, axis = axis))
    return str_arr

def open_image(fn):
    img_arr = skimage.io.imread(fn, plugin='tifffile')
    #np.array(Image.open(fn))
    img = color_scale(img_arr)
    return img

def cache_stats():
    all_files, _ = datafiles()
    imgs_arr = np.stack(open_image(fn) for fn in all_files).astype('float32')
    means = means_data(imgs_arr)
    stds = stds_data(imgs_arr)
    print("mean for the dataset is {}".format(means))
    print("Std for the dataset is {}".format(stds))
    return means,stds



def preprocess_inputs_std(x, means, stds):
    """The means and stds are train and validation base.
    It need to be train's stds and means. It might be ok since we are doing KFold split here"""
    # means = means_data(x)
    # stds = stds_data(x)
    zero_msk = (x == 0)
    # means, stds = cache_stats()
    x = np.asarray(x, dtype='float32')
    x -= means
    x /= stds
    x[zero_msk] = 0
    return x

def datafiles():
    all_files = []
    all_masks = []
    t0 = timeit.default_timer()
    imgs_folder = sys.argv[2]
    masks_folder = os.path.join(os.getcwd(),sys.argv[3])
    models_folder = os.path.join(os.getcwd(),sys.argv[4])
    if not path.isdir(models_folder):
        mkdir(models_folder)
    for f in sorted(listdir(path.join(os.getcwd(), imgs_folder))):
        if path.isfile(path.join(os.getcwd(),imgs_folder, f)) and '.tif' in f:
            img_id = f.split(img_head)[1].split('.')[0]
            all_files.append(path.join(os.getcwd(), imgs_folder, f))
            all_masks.append(path.join(masks_folder, '{0}{1}'.format(img_id, '.png')))
    all_files = np.asarray(all_files)
    all_masks = np.asarray(all_masks)
    return all_files, all_masks


def rotate_image(image, angle, scale):
    all_files,all_masks = datafiles()
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result

means, stds = cache_stats()

def batch_data_generator(train_idx, batch_size):
    all_files, all_masks = datafiles()
    inputs = []
    outputs = []
    while True:
        np.random.shuffle(train_idx)
        for i in train_idx:
            # img = skimage.io.imread(all_files[i], plugin='tifffile')
            img = open_image(all_files[i])
            if img.shape[0] != origin_shape[0]:
                img= cv2.resize(img, origin_shape)
            else:
                img = img
            if channel_no == 8:
                img = img
            else:
                band_index = rgb_index
                img = img[:, :, band_index]
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
                inputs = preprocess_inputs_std(inputs, means, stds)
                yield inputs, outputs
                inputs = []
                outputs = []

def val_data_generator(val_idx, batch_size, validation_steps):
    all_files,all_masks = datafiles()
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for i in val_idx:
            # img0 = skimage.io.imread(all_files[i], plugin='tifffile')
            img0 = open_image(all_files[i])
            if img0.shape[0] != origin_shape[0]:
                img0= cv2.resize(img0, origin_shape)
            else:
                img0 = img0
            # rgb_index = [i for i in range(channel_no)]
            if channel_no == 8:img0 = img0
            else:
                band_index = rgb_index
                img0 = img0[:, :, band_index]
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
                    inputs = preprocess_inputs_std(inputs, means, stds)
                    yield inputs, outputs
                    inputs = []
                    outputs = []
                    if step_id == validation_steps:
                        break
