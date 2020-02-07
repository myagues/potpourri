"""https://www.tensorflow.org/hub/overview"""

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


def main():
    train_data, test_data = tfds.load(
        name="imdb_reviews",
        split=["train", "test"],
        data_dir="/data/tfds",
        batch_size=-1,
        as_supervised=True,
    )
    train_examples, train_labels = tfds.as_numpy(train_data)
    test_examples, test_labels = tfds.as_numpy(test_data)

    model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(
        model, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True
    )
    # hub_layer(train_examples[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_examples,
        train_labels,
        epochs=40,
        batch_size=512,
        validation_data=(test_examples, test_labels),
        verbose=1,
    )

    results = model.evaluate(test_data, test_labels)
    print(results)


if __name__ == "__main__":
    main()
