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
from loss import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss
import skimage.io
import keras.backend as K
from data_utils import datafiles, stats_data, cache_stats, preprocess_inputs_std, rotate_image, batch_data_generator, val_data_generator
import argparse


'''
channel_no = 3
input_shape = (320, 320)
rgb_index = [0, 1, 2]
batch_size = 16
origin_shape = (325, 325)
model_id = sys.argv[1]
imgs_folder = sys.argv[2]
masks_folder = sys.argv[3]
models_folder =sys.argv[4]
all_files, all_masks = datafiles()
means, stds = cache_stats(imgs_folder)
'''


def stats_data(data):
    if len(data.shape) > 3:
        means = np.mean(data, axis = (0, 1, 2))
        stds = np.std(data, axis = (0, 1, 2))
    else:
        means = np.mean(data, axis = (0, 1))
        stds = np.std(data, axis = (0, 1))
    print(means)
    return means, stds

def color_scale(arr):
    """correct the wv-3 bands to be a composed bands of value between 0 255"""
    axis = (0, 1)
    str_arr = (arr - np.min(arr, axis = axis))*255.0/(np.max(arr, axis = axis) - np.min(arr, axis = axis))
    return str_arr

def open_image(fn):
    arr = skimage.io.imread(fn, plugin='tifffile').astype('float32')
    img = color_scale(arr)
    return img

def cache_stats(imgs_folder):
    imgs = []
    for f in listdir(path.join(imgs_folder)):
        if path.isfile(path.join(imgs_folder, f)) and '.tif' in f:
            fpath = path.join(imgs_folder, f)
            img = open_image(fpath)
            img_ = np.expand_dims(img, axis=0)
            imgs.append(img)
    imgs_arr = np.array(imgs)
    dt_means, dt_stds = stats_data(imgs_arr)
    # print("mean for the dataset is {}".format(dt_means))
    # print("Std for the dataset is {}".format(dt_stds))
    return dt_means,dt_stds

def preprocess_inputs_std(x, mean, std):
    """The means and stds are train and validation base.
    It need to be train's stds and means. It might be ok since we are doing KFold split here"""
    zero_msk = (x == 0)
    x = np.asarray(x, dtype='float32')
    x -= mean
    x /= std
    x[zero_msk] = 0
    return x

def datafiles(img_head):
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

 # = cache_stats(imgs_folder)

def batch_data_generator(train_idx, batch_size,rgb_index):
    all_files, all_masks = datafiles()
    inputs = []
    outputs = []
    while True:
        np.random.shuffle(train_idx)
        for i in train_idx:
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
                    print(inputs.shape, outputs.shape)
                    print(np.unique(inputs))
                    yield inputs, outputs
                    inputs = []
                    outputs = []
                    if step_id == validation_steps:
                        break


def train(means, stds, channel_no, rgb_index, input_shape, origin_shape):
    if model_id == 'resnet_unet':
        from resnet_unet import get_resnet_unet
        model = get_resnet_unet(input_shape, channel_no)
    elif model_id == 'inception_unet':
        from inception_unet import get_inception_resnet_v2_unet
        model = get_inception_resnet_v2_unet(input_shape, channel_no)
    elif model_id == 'linknet_unet':
        from linknet_unet import get_resnet50_linknet
        model = get_resnet50_linknet(input_shape, channel_no)
    else:
        print('No model loaded!')

    if not path.isdir(models_folder):
        mkdir(models_folder)

    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    for all_train_idx, all_val_idx in kf.split(all_files):
        train_idx = []
        val_idx = []

        for i in all_train_idx:
            train_idx.append(i)
        for i in all_val_idx:
            val_idx.append(i)

        validation_steps = int(len(val_idx) / batch_size)
        steps_per_epoch = int(len(train_idx) / batch_size)

        if validation_steps == 0 or steps_per_epoch == 0:
          continue

        print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

        np.random.seed(11)
        random.seed(11)
        tf.set_random_seed(11)


        model.compile(loss=dice_logloss3,
                    optimizer=SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        model_checkpoint = ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
        for l in model.layers:
          l.trainable = True
        model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=1e-3),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
        model.optimizer = Adam(lr=2e-4)
        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])

        np.random.seed(22)
        random.seed(22)
        tf.set_random_seed(22)
        model.load_weights(path.join(models_folder, '{}_weights.h5'.format(model_id)))
        model.compile(loss=dice_logloss,
                    optimizer=Adam(lr=5e-4),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        model_checkpoint2 = ModelCheckpoint(path.join(models_folder, '{}_weights2.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=30, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])
        optimizer=Adam(lr=1e-5)
        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=20, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])

        np.random.seed(33)
        random.seed(33)
        tf.set_random_seed(33)
        model.load_weights(path.join(models_folder, '{}_weights2.h5'.format(model_id)))
        model.compile(loss=dice_logloss2,
                    optimizer=Adam(lr=5e-5),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        model_checkpoint3 = ModelCheckpoint(path.join(models_folder, '{}_weights3.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=50, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint3])

        np.random.seed(44)
        random.seed(44)
        tf.set_random_seed(44)
        model.load_weights(path.join(models_folder, '{}_weights3.h5'.format(model_id)))
        model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=2e-5),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        model_checkpoint4 = ModelCheckpoint(path.join(models_folder, '{}_weights4.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
        model.fit_generator(generator=batch_data_generator(val_idx, batch_size, validation_steps),
                            epochs=50, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint4])
    K.clear_session()

def parse_args(args):
    desc = "ur-decoder_train"
    dhf = argparse.ArgumentDefaultsHelpFormatter
    parse0 = argparse.ArgumentParser(description = desc, formatter_class = dhf)
    parse0.add_argument('--model_id', help = 'model id is resnet_unet, inception_unet, and linknet_unet')
    parse0.add_argument('--img_dir', help='Path to satellite imagery tiles')
    parse0.add_argument('--label_dir', help='Path to label data')
    parse0.add_argument('--channel_no', default = 3, help = 'the number of bands')
    parse0.add_argument('--output_path', help = 'Path to save trained model', default = './')
    return vars(parse0.parse_args(args))

def cli():
    args = parse_args(sys.argv[1:])
    train(**args)

if __name__=="__main__":
    t0 = timeit.default_timer()
    all_files, all_masks = datafiles()
    means, stds = cache_stats(imgs_folder)
    input_shape = (320, 320)
    rgb_index = [0, 1, 2]
    batch_size = 16
    origin_shape = (325, 325)
    cli()
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
