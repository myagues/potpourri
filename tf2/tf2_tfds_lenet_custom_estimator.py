"""Example code for TF2 with LeNet and MNIST using tf.estimator
Model and parameters taken from
https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py"""

import os
import time
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_estimator.python.estimator.head.multi_class_head import \
    MultiClassHead


def preprocess_fn(feature: Dict[str, tf.Tensor],
                output_height: int = 28,
                output_width: int = 28) -> Dict[str, tf.Tensor]:
    image, label = feature['image'], feature['label']
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(
        image, output_width, output_height)
    image = tf.math.subtract(image, 128.0)
    image = tf.math.divide(image, 128.0)
    return (image, tf.cast(label, tf.int64))


def read_data(name: str, train=True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds_train, ds_test = tfds.load(name=name, split=['train', 'test'])
    ds_train = (ds_train.shuffle(1000)
                .map(preprocess_fn).batch(128).prefetch(10))
    ds_test = ds_test.map(preprocess_fn).batch(128).prefetch(10)
    return ds_train if train else ds_test


def ks_model(num_classes: int = 10,
             weight_decay: float = 0.0,
             prob: float = 0.5,
             input_shape: Tuple[int] = (28, 28, 1)) -> tf.keras.Sequential:

    weights_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation="relu",
            kernel_initializer=weights_init,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            input_shape=input_shape),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            activation="relu",
            kernel_initializer=weights_init,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            1024,
            activation="relu",
            kernel_initializer=weights_init,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        tf.keras.layers.Dropout(prob),
        tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=weights_init)])
    return model


def model_fn(features, labels, mode):

    model = ks_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.Accuracy()

    logits = model(features)
    loss_value = loss(labels, logits)
    train_vars = model.variables
    grads = tf.gradients(loss_value, train_vars)

    acc.update_state(labels, tf.math.argmax(logits, axis=1))

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss_value,
        eval_metric_ops={'acc': acc},
        train_op=optimizer.apply_gradients(zip(grads, train_vars)))
        # Tell the Estimator to save "ckpt" in an object-based format.
        # scaffold=tf_compat.train.Scaffold(saver=ckpt))
    # acc = tf.keras.metrics.Accuracy()
    # loss_metric = tf.keras.metrics.Mean()


def main():
    tf.keras.backend.clear_session()
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./estimator')

    estimator.train(input_fn=lambda: read_data('mnist'), steps=10)
    estimator.evaluate(input_fn=lambda: read_data('mnist', False), steps=10)


if __name__ == "__main__":
    main()
