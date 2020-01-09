import jax.numpy as np

from jax import lax
from jax.experimental import stax
from jax.experimental.stax import (
    Conv, Dense, Flatten, GeneralConv, MaxPool, Relu, LogSoftmax, Dropout,
    AvgPool, BatchNorm, FanInSum, FanOut, Identity
)


def lenet(num_classes: int, mode: str = "train"):
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


def pad_layer(**fun_kwargs):
    pad_size = np.sum(fun_kwargs.get("padding_config"), axis=1)
    init_fun = lambda rng, input_shape: (tuple(np.sum((input_shape, pad_size), axis=0)), ())
    apply_fun = lambda params, inputs, **kwargs: lax.pad(inputs, **fun_kwargs)
    return init_fun, apply_fun


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2 = filters
    Main = stax.serial(
        Conv(filters1, (ks, ks), strides, padding='SAME'),
        BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'),
        BatchNorm())
    Shortcut = stax.serial(
        Conv(filters2, (1, 1), strides, W_init=None),
        BatchNorm())
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
    ks = kernel_size
    filters1, filters2 = filters
    def make_main(input_shape):
        return stax.serial(
            Conv(filters1, (1, 1), padding='SAME'),
            BatchNorm(), Relu,
            Conv(filters2, (ks, ks), padding='SAME'),
            BatchNorm())
    Main = stax.shape_dependent(make_main)
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


def ResNet20(num_classes, mode: str = "train"):
    # https://github.com/google/jax/issues/139
    return stax.serial(
        pad_layer(padding_value=0.0,
                  padding_config=((0, 0, 0), (1, 1, 0), (1, 1, 0), (0, 0, 0))),
        Conv(16, (3, 3)),
        BatchNorm(), Relu,
        ConvBlock(3, [16, 16], strides=(1, 1)),
        IdentityBlock(3, [16, 16]),
        IdentityBlock(3, [16, 16]),
        ConvBlock(3, [32, 32]),
        IdentityBlock(3, [32, 32]),
        IdentityBlock(3, [32, 32]),
        ConvBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),
        AvgPool((8, 8)), Flatten, Dense(num_classes), LogSoftmax)