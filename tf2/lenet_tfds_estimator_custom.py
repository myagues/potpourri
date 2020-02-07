"""Example code for TF2 with LeNet and MNIST using tf.estimator
Model and parameters taken from https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py"""

import os
import time
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


def input_fn():
    datasets, info = tfds.load(
        name="mnist", data_dir="/data/tfds", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = datasets["train"], datasets["test"]

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label[..., tf.newaxis]

    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_data.repeat()


def ks_model(
    num_classes: int = 10,
    weight_decay: float = 0.0,
    prob: float = 0.5,
    input_shape: Tuple[int] = (28, 28, 1),
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


def model_fn(features, labels, mode):

    model = ks_model()
    training = mode == tf.estimator.ModeKeys.TRAIN

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.Accuracy(name="acc_obj")

    logits = model(features, training=training)

    reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
    loss_value = loss(labels, logits) + tf.math.add_n(reg_losses)

    accuracy = acc.update_state(y_true=labels, y_pred=tf.math.argmax(logits, axis=1))

    train_vars = model.variables
    grads = tf.gradients(loss_value, train_vars)

    train_op = None
    if training:
        # Upgrade to tf.keras.optimizers.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # Manually assign tf.compat.v1.global_step variable to optimizer.iterations
        # to make tf.compat.v1.train.global_step increased correctly.
        # This assignment is a must for any `tf.train.SessionRunHook` specified in
        # estimator, as SessionRunHooks rely on global step.
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
        # Get both the unconditional updates (the None part)
        # and the input-conditional updates (the features part).
        update_ops = model.get_updates_for(None) + model.get_updates_for(features)
        # Compute the minimize_op.
        minimize_op = optimizer.get_updates(loss_value, model.trainable_variables)[0]
        train_op = tf.group(minimize_op, *update_ops)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=logits,
        loss=loss_value,
        train_op=train_op,
        eval_metric_ops={"Accuracy": acc},
    )


def main():
    # tf.keras.backend.clear_session()
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="./estimator")

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=10000)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
