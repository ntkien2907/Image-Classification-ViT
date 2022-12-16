import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from config import *

tf.config.run_functions_eagerly(True)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def save_figures(h, path):
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(path)
    plt.close()


def load_data(path, split_ratio=0.2):
    images = shuffle(glob(os.path.join(path, '*', '*.jpg')))
    X_train, X_test = train_test_split(images, test_size=split_ratio, random_state=RANDOM_STATE)
    X_valid, X_test = train_test_split(X_test, test_size=0.5, random_state=RANDOM_STATE)
    return X_train, X_valid, X_test


def get_label(path):
    path = path.decode()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (PARAMS['IMAGE_SIZE'], PARAMS['IMAGE_SIZE']), cv2.INTER_CUBIC)
    img = img / 255.0

    patch_shape = (PARAMS['PATCH_SIZE'], PARAMS['PATCH_SIZE'], PARAMS['N_CHANNELS'])
    patches = patchify(img, patch_shape, PARAMS['PATCH_SIZE'])
    
    # patches = np.reshape(patches, (64, 25, 25, 3))
    # for i in range(patches[0]):
    #     cv2.imread(f'patched_image/{i}.png', patches[i])

    patches = np.reshape(patches, PARAMS['FLAT_PATHCHES_SHAPE'])
    patches = patches.astype(np.float32)
    
    class_name = path.split('\\')[-2]
    class_idx = PARAMS['CLASS_NAMES'].index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    return patches, class_idx


def parse(path):
    patches, labels = tf.numpy_function(get_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, PARAMS['N_CLASSES'])
    patches.set_shape(PARAMS['FLAT_PATHCHES_SHAPE'])
    labels.set_shape(PARAMS['N_CLASSES'])
    return patches, labels


def tf_dataset(images, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch_size).prefetch(8)
    return ds