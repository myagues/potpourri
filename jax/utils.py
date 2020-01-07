import math

import tensorflow as tf
import tensorflow_datasets as tfds

def get_ds_batches(
    name: str, data_dir: str, num_classes: int, split: str, batch_size: int
):
    ds, info = tfds.load(
        name=name,
        split=split,
        as_supervised=True,
        data_dir=data_dir,
        with_info=True,
    )
    ds = ds.map(
        lambda img, lbl: (
            tf.cast(img, tf.float32),
            tf.one_hot(lbl, num_classes),
        )
    )

    if split == "train":
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).prefetch(1)
    return (
        tfds.as_numpy(ds),
        math.ceil(info.splits[split].num_examples / batch_size),
        info._features["image"].shape,
        info._features["label"].num_classes,
    )
