"""
Read RGB and labels tiles to numpy ndarray.
This is developed particularly to read image and label tiles from skynet data `seattle-idaho-true`.

label_dir is the directory `seattle-idaho-true/labels/color`, and
RGB_dir is the directory 'seattle-idaho-true/images'

Usage:
          python data_from_imgs.py label_dir rgb_dir
"""

import os
import glob
import sys
from PIL import Image
from os import path as op
import numpy as np
from sklearn.model_selection import train_test_split


def open_image(fn):
    return np.array(Image.open(fn))


def read_imgs(label_path, rgb_path):
    """
    Read in rgb and label tiles;
    Stack them as a whole dataset;
    Remove the blank tile and its rgb tile together;
    Slip the whole dataset into train and tes in 80:20.

    """
    def paths(label_path, rgb_path):
        label_path = op.join(os.getcwd(), label_path)
        rgb_path = op.join(os.getcwd(), rgb_path)
        return label_path, rgb_path
    label_path, rgb_path = paths(label_path, rgb_path)
    keep_inds = list()
    fnames = sorted(os.listdir(label_path))
    img_fnames = sorted(os.listdir(rgb_path))
    lg_labels = [op.join(label_path, fn) for fn in fnames]
    for label in lg_labels:
        label_img = Image.open(label)
        label_arr = np.array(label_img)
        if label_arr.max() != 0:
            fn, _ = op.splitext(label)
            fn = fn.split("/")[-1]
            if '{}.png'.format(fn) in img_fnames:
                keep_inds.append(fn)
    rgbs = [op.join(rgb_path, '{}.png'.format(fname)) for fname in keep_inds]
    labels = [op.join(label_path, '{}.png'.format(fname)) for fname in keep_inds]
    rgbs_arr = np.stack(open_image(fn) for fn in rgbs).astype('float32')
    label_arr = np.stack(np.expand_dims(open_image(fn), 2) for fn in labels).astype('float32')
    # remove the rgb tiles have more than 10 percent black piexels
    zero_pix = np.sum(rgbs_arr.reshape(rgbs_arr.shape[0], -1) == 0, axis=1)
    keep_inds_f = zero_pix < (0.01 * rgbs_arr.shape[1] * rgbs_arr.shape[2])
    print("Loading {} of tiles to the training set.".format(len(keep_inds_f)))
    img_data = rgbs_arr[keep_inds_f, ...]
    img_masks = label_arr[keep_inds_f, ...]
    x_train, x_test, y_train, y_test = train_test_split(
        img_data, img_masks, test_size=0.2, random_state=42)
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # print(x_train.max(), x_test.max(), y_train.max(), y_test.max())
    y_train = np.where(y_train > 0, 1., 0.)
    y_test = np.where(y_test > 0, 1., 0.)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    rgb_dir = sys.argv[2]
    label_dir = sys.argv[1]
    x_train, x_test, y_train, y_test = read_imgs(label_dir, rgb_dir)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # print(y_test.max())
    road_pix = np.sum(y_train)
    print((road_pix/np.size(y_train))*100)
    dest_folder = os.getcwd()
    np.savez(op.join(dest_folder, 'data_skynet_set9_s_small.npz'),
             x_train=x_train[:, ...],
             y_train=y_train[:, ...],
             x_test=x_test[:, ...],
             y_test=y_test[:, ...])
    data = np.load('data_skynet_set9_s_small.npz')
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print("final data shapes are... ")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
