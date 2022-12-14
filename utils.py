import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from config import *


np.random.seed(RANDOM_STATE)


def load_data(path, split_ratio=0.4):
    images = shuffle(glob(os.path.join(path, '*', '*.jpg')))
    X_train, X_test = train_test_split(images, test_size=split_ratio, random_state=RANDOM_STATE)
    X_val, X_test = train_test_split(X_train, test_size=0.5, random_state=RANDOM_STATE)
    return X_train, X_val, X_test


def get_label(path):
    path = path.decode()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0

    patch_shape = (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
    patches = patchify(img, patch_shape, PATCH_SIZE)
    
    # patches = np.reshape(patches, (64, 25, 25, 3))
    # for i in range(patches[0]):
    #     cv2.imread(f'patched_image/{i}.png', patches[i])

    patches = np.reshape(patches, FLAT_PATHCHES_SHAPE)
    patches = patches.astype(np.float32)
    
    class_name = path.split('\\')[-2]
    class_idx = CLASS_NAMES.index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    return patches, class_idx


def parse(path):
    patches, labels = tf.numpy_function(get_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, N_CLASSES)
    patches.set_shape(FLAT_PATHCHES_SHAPE)
    labels.set_shape(N_CLASSES)
    return patches, labels


def tf_dataset(images, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch_size).prefetch(8)
    return ds