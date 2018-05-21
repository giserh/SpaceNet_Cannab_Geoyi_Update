# -*- coding: utf-8 -*-
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
from models import get_vgg_unet
import skimage.io
from tqdm import tqdm

input_shape = (1344, 1344)

def preprocess_inputs(x):
    zero_msk = (x == 0)
    x = x / 8.0
    x -= 127.5
    x[zero_msk] = 0
    return x

models_folder = '/wdata/nn_models'
pred_folder = '/wdata/predictions'
model_name = 'vgg_big'

cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

ignored_cities = [0]

if __name__ == '__main__':
    models_folder = 'wdata/AOI_3_Paris_Roads_Train/nn_models'
    pred_folder = 'wdata/predictions'
    model_name = 'vgg_big'

    city_datasets = dict(Vegas = 'AOI_2_Vegas_Roads_Train',
                         Paris = 'AOI_3_Paris_Roads_Train')
    cities = city_datasets.values()

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
            # if i in ignored_cities or not path.isfile(path.join(models_folder, 'vgg_model3_weights2_{0}_{1}.h5'.format(cities[i], it))):
            #     models.append(None)
            #     continue
            if not path.isdir(path.join(path.join(pred_folder, model_name, str(it), city))):
                mkdir(path.join(path.join(pred_folder, model_name, str(it), city)))
            model = get_vgg_unet(input_shape, weights=None)
            model.load_weights(path.join(models_folder, 'vgg_model3_weights2_{0}_{1}.h5'.format(city, it)))
            models.append(model)

        print('Predictiong fold', it)
        for city, d in city_datasets.items():
            for f in tqdm(sorted(listdir(path.join('wdata',d, 'MUL-PanSharpen')))):
                if path.isfile(path.join('wdata',d, 'MUL-PanSharpen', f)) and '.tif' in f:
                    img_id = f.split('PanSharpen_')[1].split('.')[0]
                    cinp = np.zeros((4,))
                    cinp[cities.index(img_id.split('_')[2])] = 1.0
                    cid = cinp.argmax()
                    # if cid in ignored_cities:
                    #     continue
                    fpath = path.join('wdata',d, 'MUL-PanSharpen', f)
                    img = skimage.io.imread(fpath, plugin='tifffile')
                    img = cv2.copyMakeBorder(img, 22, 22, 22, 22, cv2.BORDER_REFLECT_101)
                    inp = []
                    inp.append(img)
                    inp.append(np.rot90(img, k=1))
                    inp = np.asarray(inp)
                    inp = preprocess_inputs(inp)
                    inp2 = []
                    inp2.append(cinp)
                    inp2.append(cinp)
                    inp2 = np.asarray(inp2)
                    pred = models[cid].predict([inp, inp2])
                    mask = pred[0] + np.rot90(pred[1], k=3)
                    mask /= 2
                    mask = mask[22:1322, 22:1322, ...]
                    mask = mask * 255
                    mask = mask.astype('uint8')
                    cv2.imwrite(path.join(pred_folder, model_name, str(it), city, '{0}.png'.format(img_id)), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
