"""https://www.tensorflow.org/hub/overview"""

import itertools
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from typing import Dict, Tuple


def preprocess_fn(
    feature: Dict[str, tf.Tensor],
    output_height: int = 224,
    output_width: int = 224,
    training: str = False,
) -> Dict[str, tf.Tensor]:
    image, label = feature["image"], feature["label"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(tf.cast(label, tf.int64), 5)
    image = tf.image.resize(image, [output_width, output_height])
    if training:
        image = tf.image.random_flip_left_right(image)
    return image, label


def main():
    train_ds, test_ds = tfds.load(
        name="tf_flowers",
        split=["train[:80%]", "train[-20%:]"],
        data_dir="/data/tfds",
        shuffle_files=True,
    )

    map_fn = lambda x: preprocess_fn(x, training=False)
    train_ds = train_ds.shuffle(1000).map(map_fn).batch(32).prefetch(10)
    test_ds = test_ds.shuffle(1000).map(preprocess_fn).batch(32).prefetch(10)

    model = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    hub_layer = hub.KerasLayer(model, output_shape=[1280], trainable=True)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(rate=0.2)),
    model.add(
        tf.keras.layers.Dense(
            5,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        )
    )
    model.build([None, 224, 224, 3])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, epochs=2, validation_data=test_ds, verbose=1,
    )

    saved_model_path = "/tmp/saved_flowers_model"
    tf.saved_model.save(model, saved_model_path)

    optimize_lite_model = False  # @param {type:"boolean"}
    num_calibration_examples = 60  # @param {type:"slider", min:0, max:1000, step:1}
    representative_dataset = None
    if optimize_lite_model and num_calibration_examples:
        # Use a bounded number of training examples without labels for calibration.
        # TFLiteConverter expects a list of input tensors, each with batch size 1.
        representative_dataset = lambda _: itertools.islice(
            ([image[None, ...]] for batch, _ in iter(test_ds) for image in batch),
            num_calibration_examples,
        )

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    if optimize_lite_model:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset:  # This is optional, see above.
            converter.representative_dataset = representative_dataset
    lite_model_content = converter.convert()

    with open("/tmp/lite_flowers_model", "wb") as f:
        f.write(lite_model_content)

    print(f"Wrote TFLite model of {len(lite_model_content):d} bytes.")

    interpreter = tf.lite.Interpreter(model_content=lite_model_content)
    # This little helper wraps the TF Lite interpreter as a numpy-to-numpy function.
    def lite_model(images):
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], images)
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

    num_eval_examples = 50  # @param {type:"slider", min:0, max:700}
    eval_dataset = (
        (image, label)  # TFLite expects batch size 1.
        for batch in iter(test_ds)
        for (image, label) in zip(*batch)
    )
    count = 0
    count_lite_tf_agree = 0
    count_lite_correct = 0
    for image, label in eval_dataset:
        probs_lite = lite_model(image[None, ...])[0]
        probs_tf = model(image[None, ...]).numpy()[0]
        y_lite = np.argmax(probs_lite)
        y_tf = np.argmax(probs_tf)
        y_true = np.argmax(label)
        count += 1
        if y_lite == y_tf:
            count_lite_tf_agree += 1
        if y_lite == y_true:
            count_lite_correct += 1
        if count >= num_eval_examples:
            break
    print(
        f"TF Lite model agrees with original model on {count_lite_tf_agree:d} of "
        f"{count:d} examples ({(100.0 * count_lite_tf_agree / count):g}%)."
    )
    print(
        f"TF Lite model is accurate on {count_lite_correct:d} of {count:d} examples "
        f"({(100.0 * count_lite_correct / count):g}%)."
    )


if __name__ == "__main__":
    main()
