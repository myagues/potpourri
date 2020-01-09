import os
import math

import tensorflow as tf

from tensorflow import keras

def get_ds_keras(ds_name:str, ds_dir: str, batch_size: int, num_classes:int):
    if ds_name == "mnist":
        ds_cache = os.path.join(ds_dir, "mnist.npz")
        ds_path = ds_cache if os.path.exists(ds_cache) else ""

        ds = keras.datasets.mnist
        input_shape = (batch_size, 28, 28, 1)
        lambda_fn = lambda img, lbl: (
            tf.expand_dims(img, axis=-1),
            tf.one_hot(lbl, num_classes),
        )

        (train_images, train_labels), (test_images, test_labels) = \
            ds.load_data(path=ds_path)
    
    elif ds_name == "cifar10":
        ds = keras.datasets.cifar10
        input_shape = (batch_size, 32, 32, 3)
        lambda_fn = lambda img, lbl: (img, tf.one_hot(tf.squeeze(lbl), num_classes))
        (train_images, train_labels), (test_images, test_labels) = ds.load_data()
    
    else:
        print("'ds_name' not recognized!!!")

    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_ds = train_ds.map(lambda_fn)
    test_ds = test_ds.map(lambda_fn)

    train_ds = train_ds.cache().shuffle(10000).batch(batch_size).prefetch(10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(10)
    return (train_ds, test_ds, input_shape)