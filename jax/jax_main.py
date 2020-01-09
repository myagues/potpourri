import collections
import math
import time

import jax.numpy as np

from jax import jit, grad, random
from jax.experimental import optimizers
from jax_nets import lenet, ResNet20
from utils import get_ds_keras


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
    ds_name = "mnist"
    ds_dir = "/data"
    # net_name = "ResNet20"
    num_epochs = 15
    batch_size = 128
    step_size = 1e-3
    num_classes = 10
    net_dict = {"mnist": lenet, "cifar10": ResNet20}
    net = net_dict[ds_name]

    train_ds, test_ds, input_shape = \
        get_ds_keras(ds_name, ds_dir, batch_size, num_classes)
    
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    
    init_train_fun, train_fun = net(num_classes)
    _, init_params = init_train_fun(key, input_shape)
    opt_state = opt_init(init_params)

    _, test_fun = net(num_classes, mode="test")

    update_step = jit_update_fun(train_fun, loss_fun, (opt_update, get_params))
    predict_step = jit_predict_fun(test_fun)

    step = 0
    for ep in range(num_epochs):
        start_time = time.time()
        for i, batch in enumerate(train_ds):
            key, subkey = random.split(key)
            batch = (batch[0].numpy(), batch[1].numpy())
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
        start_time = time.time()
        for i, batch in enumerate(test_ds):
            batch = (batch[0].numpy(), batch[1].numpy())
            inputs, labels = batch
            key, subkey = random.split(key)
            logits = predict_step(trained_params, inputs, rng=subkey)
            test_metrics["loss"] += cross_entropy(logits, labels)
            test_metrics["acc"] += accuracy(logits, labels)

        print(
            f"Epoch: {ep:d}  "
            f"Time/Step: {(time.time() - start_time) / (i + 1):.3f}s  "
            f"Eval Loss: {test_metrics['loss'] / (i + 1):.5f}  "
            f"Eval Acc: {test_metrics['acc'] / (i + 1):.3f}"
        )

if __name__ == "__main__":
    main()
