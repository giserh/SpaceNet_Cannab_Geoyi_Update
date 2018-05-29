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
from data_utils import datafiles, means_data, stds_data, preprocess_inputs_std, rotate_image, batch_data_generator, val_data_generator

#from other part of data utils improt all_files, all_pan_files, all_masks, city_id

channel_no = 3
input_shape = (320, 320)
cities = ['Vegas','Paris']
city_datasets = dict(Vegas = 'AOI_2_Vegas_Roads_Train',
                     Paris = 'AOI_3_Paris_Roads_Train')
city_id = -1
batch_size = 16
it = -1
all_files, all_pan_files, all_city_inp, all_masks = datafiles(cities, city_datasets)
fold_nums = [0, 1]

d = 'AOI_3_Paris_Roads_Train'
masks_folder = os.path.join(os.getcwd(),'wdata/{}/masks_smallest'.format(d))
models_folder = os.path.join(os.getcwd(),'wdata/{}/nn_models'.format(d))

means = [[290.42, 446.84, 591.88], [178.33, 260.14, 287.4]]
stds = [[75.42, 177.98, 288.81], [16.4, 45.69, 79.42]]

t0 = timeit.default_timer()

model_id = sys.argv[1]
if model_id == 'resnet_unet':
  from resnet_unet import get_resnet_unet
  model = get_resnet_unet(input_shape)
else:
  from inception_unet import get_inception_resnet_v2_unet
  model = get_inception_resnet_v2_unet(input_shape)

if not path.isdir(models_folder):
    mkdir(models_folder)

kf = KFold(n_splits=4, shuffle=True, random_state=1)
for all_train_idx, all_val_idx in kf.split(all_files):
  it += 1

  if it not in fold_nums:
      continue

  for cid, city_ in enumerate(cities):
      city_id = cid
      train_idx = []
      val_idx = []

      for i in all_train_idx:
          if all_city_inp[i][city_id] == 1:
              train_idx.append(i)
      for i in all_val_idx:
          if all_city_inp[i][city_id] == 1:
              val_idx.append(i)

      validation_steps = int(len(val_idx) / batch_size)
      steps_per_epoch = int(len(train_idx) / batch_size)

      if validation_steps == 0 or steps_per_epoch == 0:
          print("No data for city", cities[city_id])
          continue

      print('Training city', cities[city_id], 'fold', it)
      print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

      np.random.seed(it+11)
      random.seed(it+11)
      tf.set_random_seed(it+11)

      print('Training model', it, cities[city_id])

      # models = dict(resnet_unet = get_resnet_unet(input_shape), inception_unet = get_inception_resnet_v2_unet(input_shape))


      model.compile(loss=dice_logloss3,
                    optimizer=SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

      model_checkpoint = ModelCheckpoint(path.join(models_folder, '{}_weights_{}_{}.h5'.format(model_id, cities[city_id], it)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=15, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
      for l in model.layers:
          l.trainable = True
      model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=1e-3),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=30, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])
      model.optimizer = Adam(lr=2e-4)
      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=15, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint])

      np.random.seed(it+22)
      random.seed(it+22)
      tf.set_random_seed(it+22)
      model.load_weights(path.join(models_folder, '{}_weights_{}_{}.h5'.format(model_id, cities[city_id], it)))
      model.compile(loss=dice_logloss,
                    optimizer=Adam(lr=5e-4),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
      model_checkpoint2 = ModelCheckpoint(path.join(models_folder, '{}_weights2_{}_{}.h5'.format(model_id, cities[city_id], it)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=20, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])
      optimizer=Adam(lr=1e-5)
      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=10, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint2])

      np.random.seed(it+33)
      random.seed(it+33)
      tf.set_random_seed(it+33)
      model.load_weights(path.join(models_folder, '{}_weights2_{0}_{1}.h5'.format(model_id, cities[city_id], it)))
      model.compile(loss=dice_logloss2,
                    optimizer=Adam(lr=5e-5),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
      model_checkpoint3 = ModelCheckpoint(path.join(models_folder, '{}_weights3_{}_{}.h5'.format(model_id, cities[city_id], it)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint3])

      np.random.seed(it+44)
      random.seed(it+44)
      tf.set_random_seed(it+44)
      model.load_weights(path.join(models_folder, '{}_weights3_{}_{}.h5'.format(model_id, cities[city_id], it)))
      model.compile(loss=dice_logloss3,
                    optimizer=Adam(lr=2e-5),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
      model_checkpoint4 = ModelCheckpoint(path.join(models_folder, '{}_weights4_{}_{}.h5'.format(model_id, cities[city_id], it)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=True, mode='max')
      model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                            epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint4])
      K.clear_session()

elapsed = timeit.default_timer() - t0
print('Time: {:.3f} min'.format(elapsed / 60))
