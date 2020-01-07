import collections
import time

import jax.numpy as np
import numpy.random as npr
import tensorflow as tf

from jax import jit, grad, random
from jax.experimental import optimizers, stax
from jax.experimental.stax import (
    Conv,
    Dense,
    Flatten,
    GeneralConv,
    MaxPool,
    Relu,
    LogSoftmax,
    Dropout,
)
from tqdm import tqdm
from utils import get_ds_batches


def lenet(num_classes: int, mode: str = "train") -> stax:
    return stax.serial(
        # GeneralConv(('NHWC', 'OIHW', 'NHWC'), 32, (5, 5)), Relu,
        Conv(32, (5, 5)),
        Relu,
        MaxPool((2, 2), strides=(2, 2)),
        Conv(64, (5, 5)),
        Relu,
        MaxPool((2, 2), strides=(2, 2)),
        Flatten,
        Dense(1024),
        Relu,
        Dropout(0.5, mode),
        Dense(num_classes),
        LogSoftmax,
    )


def cross_entropy(logits, labels):
    return -np.mean(np.sum(logits * labels, axis=-1))


def loss_fun(params, batch, predict_fun, rng=None):
    inputs, labels = batch
    logits = predict_fun(params, inputs, rng=rng)
    return cross_entropy(logits, labels)


def accuracy(logits, labels):
    predicted_class = np.argmax(logits, axis=1)
    labels_class = np.argmax(labels, axis=1)
    return np.mean(predicted_class == labels_class)


def jit_update_fun(model_fun, loss, opt):
    opt_update, get_params = opt

    def update(i, opt_state, batch, rng):
        params = get_params(opt_state)
        grads = grad(loss_fun)(params, batch, model_fun, rng)
        return opt_update(i, grads, opt_state)

    return jit(update)


def jit_predict_fun(model_fun):
    def predict(params, inputs, rng=None):
        return jit(model_fun)(params, inputs, rng=rng)

    return predict


def main():
    key = random.PRNGKey(0)
    num_epochs = 5
    batch_size = 128
    step_size = 1e-3
    # data_dir = 'projects/tfds'

    train_ds = get_ds_batches("mnist", data_dir, 10, "train", batch_size)
    _, train_len, img_shape, num_classes = train_ds
    test_ds = get_ds_batches("mnist", data_dir, 10, "test", batch_size)
    _, test_len, _, _ = test_ds
    input_shape = (batch_size,) + img_shape

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)

    init_train_fun, train_fun = lenet(num_classes)
    _, init_params = init_train_fun(key, input_shape)
    opt_state = opt_init(init_params)

    _, test_fun = lenet(num_classes, mode="test")

    update_step = jit_update_fun(train_fun, loss_fun, (opt_update, get_params))
    predict_step = jit_predict_fun(test_fun)

    step = 0
    for ep in range(num_epochs):
        train_batches, _, _, _ = get_ds_batches(
            "mnist", data_dir, 10, "train", batch_size
        )
        start_time = time.time()
        for i, batch in tqdm(enumerate(train_batches), total=train_len):
            key, subkey = random.split(key)
            opt_state = update_step(step, opt_state, batch, subkey)
            if step % 100 == 0:
                inputs, labels = batch
                logits = predict_step(get_params(opt_state), inputs, rng=subkey)
                print(
                    f"Step: {step:d},  "
                    f"Time/Step: {(time.time() - start_time) / (i + 1):.3f}s  "
                    f"Loss: {cross_entropy(logits, labels):.5f}  "
                    f"Acc: {accuracy(logits, labels):.3f}"
                )
            step += 1
        trained_params = get_params(opt_state)

        test_metrics = collections.defaultdict(float)
        test_batches, _, _, _ = get_ds_batches(
            "mnist", data_dir, 10, "test", batch_size
        )
        start_time = time.time()
        for i, batch in tqdm(enumerate(test_batches), total=test_len):
            inputs, labels = batch
            key, subkey = random.split(key)
            logits = predict_step(trained_params, inputs, rng=subkey)
            test_metrics["loss"] += cross_entropy(logits, labels)
            test_metrics["acc"] += accuracy(logits, labels)

        print(
            f"Epoch: {ep:d}  "
            f"Time/Step: {(time.time() - start_time) / (i + 1):.3f}s  "
            f"Eval Loss: {test_metrics['loss'] / test_len:.5f}  "
            f"Eval Acc: {test_metrics['acc'] / test_len:.3f}"
        )


if __name__ == "__main__":
    main()
