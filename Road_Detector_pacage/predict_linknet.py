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
from linknet import get_resnet50_linknet
import skimage.io
from tqdm import tqdm

input_shape = (1344, 1344)

means = [[290.42, 446.84, 591.88, 442.45, 424.66, 418.13, 554.13, 354.34, 566.86],
         [178.33, 260.14, 287.4, 161.44, 211.46, 198.83, 453.27, 228.99, 242.67],
         [357.82, 344.64, 436.76, 452.17, 290.35, 439.7, 440.43, 393.6, 452.5],
         [386.98, 415.74, 601.29, 755.34, 527.79, 729.95, 641, 611.41, 697.17]]
stds = [[75.42, 177.98, 288.81, 250.24, 260.55, 220.09, 299.67, 191.47, 285.25],
        [16.4, 45.69, 79.42, 61.91, 99.64, 81.17, 210.34, 106.31, 80.89],
        [35.23, 58, 89.42, 115.7, 90.45, 109.5, 144.61, 136.77, 99.11],
        [37.9, 59.95, 99.56, 131.14, 96.26, 107.79, 98.77, 92.2, 107.9]]
def preprocess_inputs_std(x, city_id):
    zero_msk = (x == 0)
    x = np.asarray(x, dtype='float32')
    for i in range(9):
        x[..., i] -= means[city_id][i]
        x[..., i] /= stds[city_id][i]
    x[zero_msk] = 0
    return x



# ignored_cities = [1, 2]

if __name__ == '__main__':
    models_folder = 'wdata/AOI_3_Paris_Roads_Train/nn_models'
    pred_folder = 'wdata/predictions'
    model_name = 'linknet_big'

    city_datasets = dict(Vegas = 'AOI_2_Vegas_Roads_Train',
                         Paris = 'AOI_3_Paris_Roads_Train')
    # cities = city_datasets.values()
    # cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']
    cities = ['Vegas', 'Paris']

    t0 = timeit.default_timer()

    # test_folders = []
    #
    # for i in range(1, len(sys.argv) - 1):
    #     test_folders.append(sys.argv[i])

    if not path.isdir(pred_folder):
        mkdir(os.path.join(os.getcwd(),pred_folder))

    if not path.isdir(path.join(pred_folder, model_name)):
        mkdir(path.join(pred_folder, model_name))

    for it in [0, 1]:
        models = []

        if not path.isdir(path.join(pred_folder, model_name, str(it))):
            mkdir(path.join(pred_folder, model_name, str(it)))

        for i, city in enumerate(city_datasets):
            # if i in ignored_cities or not path.isfile(path.join(models_folder, 'linknet_big_model_weights4_{0}_{1}.h5'.format(cities[i], it))):
            #     models.append(None)
            #     continue
            if not path.isdir(path.join(path.join(pred_folder, model_name, str(it), cities[i]))):
                mkdir(path.join(path.join(pred_folder, model_name, str(it), cities[i])))
            model = get_resnet50_linknet(input_shape, weights=None)
            model.load_weights(path.join(models_folder, 'linknet_big_model_weights4_{0}_{1}.h5'.format(cities[i], it)))
            models.append(model)

        print('Predictiong fold', it)
        for city, d in city_datasets.items():
            for f in tqdm(sorted(listdir(path.join('wdata', d, 'MUL-PanSharpen')))):
                if path.isfile(path.join('wdata', d, 'MUL-PanSharpen', f)) and '.tif' in f:
                    img_id = f.split('PanSharpen_')[1].split('.')[0]
                    cinp = np.zeros((4,))
                    cinp[cities.index(img_id.split('_')[2])] = 1.0
                    cid = cinp.argmax()
                    fpath = path.join('wdata', d, 'MUL-PanSharpen', f)
                    img = skimage.io.imread(fpath, plugin='tifffile')
                    pan = skimage.io.imread(path.join('wdata', d, 'PAN', 'PAN_{0}.tif'.format(img_id)), plugin='tifffile')
                    pan = pan[..., np.newaxis]
                    img = np.concatenate([img, pan], axis=2)
                    img = cv2.copyMakeBorder(img, 22, 22, 22, 22, cv2.BORDER_REFLECT_101)
                    inp = []
                    inp.append(img)
                    inp.append(np.rot90(img, k=1))
                    inp = np.asarray(inp)
                    inp = preprocess_inputs_std(inp, cid)
                    pred = models[cid].predict(inp)
                    mask = pred[0] + np.rot90(pred[1], k=3)
                    mask /= 2
                    mask = mask[22:1322, 22:1322, ...]
                    mask = mask * 255
                    mask = mask.astype('uint8')
                    cv2.imwrite(path.join(pred_folder, model_name, str(it), city, '{0}.png'.format(img_id)), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
