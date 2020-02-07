"""Example code for TF2 with LeNet and MNIST.
Model and parameters taken from
https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py"""

import os
import time
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_fn(
    feature: Dict[str, tf.Tensor], output_height: int = 28, output_width: int = 28
) -> Dict[str, tf.Tensor]:
    image, label = feature["image"], feature["label"]
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int64)
    image = tf.image.resize_with_crop_or_pad(image, output_width, output_height)
    image = tf.math.subtract(image, 128.0)
    image = tf.math.divide(image, 128.0)
    return {"image": image, "label": label}


def read_data(name: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds_train, ds_test = tfds.load(
        name=name, split=["train", "test"], data_dir="/data/tfds"
    )
    ds_train = ds_train.shuffle(1000).map(preprocess_fn).batch(128).prefetch(10)
    ds_test = ds_test.map(preprocess_fn).batch(128).prefetch(10)
    return (ds_train, ds_test)


def ks_model(
    num_classes: int = 10,
    weight_decay: float = 0.0,
    prob: float = 0.5,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
) -> tf.keras.Sequential:

    weights_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=5,
                activation="relu",
                kernel_initializer=weights_init,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=5,
                activation="relu",
                kernel_initializer=weights_init,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                1024,
                activation="relu",
                kernel_initializer=weights_init,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            ),
            tf.keras.layers.Dropout(prob),
            tf.keras.layers.Dense(num_classes, kernel_initializer=weights_init),
        ]
    )
    return model


def main():

    num_epochs = 10

    ckpt_dir = "./ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt")

    ds_train, ds_test = read_data(name="mnist")
    model = ks_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    acc = tf.keras.metrics.Accuracy()
    loss_metric = tf.keras.metrics.Mean()
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))

    for ep in range(num_epochs):
        for idx, features in enumerate(ds_train):
            start_step = time.time()
            images, labels = features["image"], features["label"]

            # with tf.GradientTape() as tape:
            #    logits = model(images)
            #    loss_value = loss(labels, logits)
            # grads = tape.gradient(loss_value, model.variables)
            # optimizer.apply_gradients(zip(grads, model.variables))
            logits = model(images)
            loss_value = loss(labels, logits)
            optimizer.minimize(
                lambda: loss(labels, model(images)), var_list=model.trainable_variables
            )

            acc.update_state(labels, tf.math.argmax(logits, axis=1))
            loss_metric(loss_value)

            if idx % 50 == 0:
                print(
                    f"Epoch: {ep:2d}\t"
                    f"Step: {idx:4d}\t"
                    f"Train Loss: {loss_value.numpy():0.5f}\t"
                    f"Train Acc: {acc.result().numpy():0.5f}\t"
                    f"Step Time: {time.time() - start_step:0.5f}s"
                )
        ckpt.save(ckpt_prefix)

        acc.reset_states()
        loss_metric.reset_states()
        for idx, features in enumerate(ds_test):
            images, labels = features["image"], features["label"]
            logits = model(images, training=False)
            acc.update_state(labels, tf.math.argmax(logits, axis=1))
            loss_metric.update_state(loss(labels, logits))
        print(
            f"Epoch: {ep:2d}\t"
            f"Test Loss: {loss_metric.result().numpy():0.5f}\t"
            f"Test Acc: {acc.result().numpy():0.5f}"
        )
        acc.reset_states()
        loss_metric.reset_states()


if __name__ == "__main__":
    main()
