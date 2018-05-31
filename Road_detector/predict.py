# -*- coding: utf-8 -*-
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
import cv2
# from models import get_resnet_unet
import skimage.io
from tqdm import tqdm
from data_utils import means_data, stds_data, preprocess_inputs_std, datafiles, cache_stats

input_shape = (352, 352)
# input_shape for the prediction is inputshape[0] + 14 + 13
origin_shape = (325, 325)
border = (13, 14)

channel_no = 3
img_head = 'RGB-PanSharpen_'
rgb_index = [0, 1, 2]
all_files, _ =datafiles()
means, stds = cache_stats(all_files)
pred_folder = 'wdata/predictions'

model_name = 'resnet_NEW_TRAIN'
model_id = sys.argv[1]
imgs_folder = sys.argv[2]
masks_folder = sys.argv[3]
models_folder =sys.argv[4]
# means = [[290.42, 446.84, 591.88], [178.33, 260.14, 287.4]]
# stds = [[75.42, 177.98, 288.81], [16.4, 45.69, 79.42]]




# ignored_cities = [0, 3]

if __name__ == '__main__':

    # city_datasets = dict(Vegas = 'AOI_2_Vegas_Roads_Test_Public',
    #                      Paris = 'AOI_3_Paris_Roads_Test_Public')
    #
    # cities = ['Vegas', 'Paris']
    # cities = city_datasets.values()
    # models = dict(resnet_unet = get_resnet_unet(input_shape, weights=None), inception_unet = get_inception_resnet_v2_unet(input_shape, weights=None))

    t0 = timeit.default_timer()
    #
    # test_folders = []
    #
    # for i in range(1, len(sys.argv) - 1):
    #     test_folders.append(sys.argv[i])

    if not path.isdir(pred_folder):
        mkdir(os.path.join(os.getcwd(),pred_folder))

    if not path.isdir(path.join(pred_folder, model_name)):
        mkdir(path.join(pred_folder, model_name))

    # for it in [0, 1]:
    # models = []

    if not path.isdir(path.join(pred_folder, model_name)):
        mkdir(path.join(pred_folder, model_name))

    # for i, city in enumerate(city_datasets):
        # if i in ignored_cities or not path.isfile(path.join(models_folder, 'resnet_smallest_model_weights4_{0}_{1}.h5'.format(cities[i], it))):
        #     models.append(None)
        #     continue
    if not path.isdir(path.join(path.join(pred_folder, model_name))):
        mkdir(path.join(path.join(pred_folder, model_name)))
    if model_id == 'resnet_unet':
        from resnet_unet import get_resnet_unet
        model = get_resnet_unet(input_shape, channel_no)
        # model.load_weights(path.join(models_folder, '{}_weights4_{0}.h5'.format(model_id, it)))
        # models.append(model)
    else:
        from inception_unet import get_inception_resnet_v2_unet
        model = get_inception_resnet_v2_unet(input_shape, channel_no)
    model.load_weights(path.join(models_folder, '{}_weights4.h5'.format(model_id)))

    if not path.isdir(models_folder):
        mkdir(models_folder)
    # model = get_resnet_unet(input_shape, weights=None)
    # model = models[model_id]


    # print('Predictiong fold', it)
    # for city, d in city_datasets.items():
    for f in tqdm(sorted(listdir(path.join(imgs_folder)))):
        if path.isfile(path.join(imgs_folder, f)) and '.tif' in f:
            img_id = f.split(img_head)[1].split('.')[0]
            # cinp = np.zeros((4,))
            # cinp[cities.index(img_id.split('_')[2])] = 1.0
            # cid = cinp.argmax()

            fpath = path.join(imgs_folder, f)
            img = open_image(fpath)
            # skimage.io.imread(fpath, plugin='tifffile')
            if img.shape[0] != origin_shape[0]:
                img= cv2.resize(img, origin_shape)
            else:
                img = img

            # pan = skimage.io.imread(path.join('wdata', 'AOI_3_Paris_Roads_Test_Public', 'PAN', 'PAN_{0}.tif'.format(img_id)), plugin='tifffile')
            # pan = cv2.resize(pan, (325, 325))
            # pan = pan[..., np.newaxis]
            # img = np.concatenate([img, pan], axis=2)
            if channel_no == 8:
                img = img
            else:
                band_index = rgb_index
                img = img[:, :, band_index]
            img = cv2.copyMakeBorder(img, 13, 14, 13, 14, cv2.BORDER_REFLECT_101)
            inp = []
            inp.append(img)
            inp.append(np.rot90(img, k=1))
            inp = np.asarray(inp)
            inp = preprocess_inputs_std(inp, means, stds)
            pred = model.predict(inp)
            mask = pred[0] + np.rot90(pred[1], k=3)
            mask /= 2
            mask = mask[13:338, 13:338, ...]
            mask = mask * 255
            mask = mask.astype('uint8')
            cv2.imwrite(path.join(pred_folder, model_name,'{}.png'.format(img_id)), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

elapsed = timeit.default_timer() - t0
print('Time: {:.3f} min'.format(elapsed / 60))
