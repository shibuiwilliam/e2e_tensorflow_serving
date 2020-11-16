import json
import numpy as np
from typing import Dict, List, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_dataset(file_path: str = "dataset.json") -> Dict[str, Any]:
    with open(file_path, "r") as f:
        dataset = json.load(f)
    return dataset


def make_label(target: List[str]) -> List[str]:
    return list(set(target))


def make_text_vectorizer(
    data: np.ndarray,
) -> tf.keras.layers.experimental.preprocessing.TextVectorization:
    text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        output_mode="tf-idf", ngrams=2
    )
    text_vectorizer.adapt(data)
    return text_vectorizer


def define_model(
    text_vectorizer: tf.keras.layers.experimental.preprocessing.TextVectorization,
    optimizer: str = "adam",
    loss: str = "categorical_crossentropy",
    metrics: List[str] = ["accuracy"],
) -> tf.keras.Model:
    inputs = keras.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = layers.Dense(1)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(6, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


class TextModel(tf.keras.Model):
    def __init__(self, model: tf.keras.Model, labels: List[str]):
        super().__init__(self)
        self.model = model
        self.labels = labels

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="text")]
    )
    def serving_fn(self, text: str) -> tf.Tensor:
        predictions = self.model(text)

        def _convert_to_label(candidates):
            max_prob = tf.math.reduce_max(candidates)
            idx = tf.where(tf.equal(candidates, max_prob))
            label = tf.squeeze(tf.gather(self.labels, idx))
            return label

        return tf.map_fn(_convert_to_label, predictions, dtype=tf.string)

    def save(self, export_path="./saved_model/text/"):
        signatures = {"serving_default": self.serving_fn}
        tf.keras.backend.set_learning_phase(0)
        tf.saved_model.save(self, export_path, signatures=signatures)


def main():
    dataset = load_dataset(file_path="dataset.json")
    data = dataset["data"]
    target = dataset["target"]
    labels = make_label(target=target)
    label_dict = {t: i for i, t in enumerate(labels)}
    target_int = [label_dict[t] for t in target]
    train_data = np.array(data)
    train_target = tf.keras.utils.to_categorical(target_int)

    text_vectorizer = make_text_vectorizer(data=train_data)
    model = define_model(text_vectorizer=text_vectorizer)
    model.fit(train_data, train_target, epochs=40, batch_size=16)

    text_model = TextModel(model=model, labels=labels)
    version_number = 0
    text_model.save(f"./saved_model/text/{version_number}/")


if __name__ == "__main__":
    main()
