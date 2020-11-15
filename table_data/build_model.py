import numpy as np
import pandas as pd
from typing import Tuple, List, Any
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def download_data(
    url: str = "https://storage.googleapis.com/applied-dl/heart.csv",
) -> pd.DataFrame:
    dataframe = pd.read_csv(url)
    return dataframe


def split_data(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    return train, val, test


def df_to_dataset(
    dataframe: pd.DataFrame, shuffle: bool = True, batch_size: int = 16
) -> tf.data.Dataset:
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def define_model(
    feature_columns: List[Any],
    optimizer: str = "adam",
    loss: str = "binary_crossentropy",
    metrics: List[str] = ["accuracy"],
) -> tf.keras.Model:
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.Sequential(
        [
            feature_layer,
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def save_model(model: tf.keras.Model, version_number: int = 0):
    save_model_path = f"./saved_model/table_data/{version_number}"
    model.save(save_model_path)
    print(f"saved model to {save_model_path}")


def main():
    batch_size = 16
    dataframe = download_data()
    train, val, test = split_data(dataframe=dataframe)
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    feature_columns = []

    for header in ["age", "trestbps", "chol", "thalach", "oldpeak", "slope", "ca"]:
        feature_columns.append(feature_column.numeric_column(header))

    age = feature_column.numeric_column("age")
    age_buckets = feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    )
    feature_columns.append(age_buckets)

    thal = feature_column.categorical_column_with_vocabulary_list(
        "thal", ["fixed", "normal", "reversible"]
    )
    thal_one_hot = feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    thal_embedding = feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    crossed_feature = feature_column.crossed_column(
        [age_buckets, thal], hash_bucket_size=1000
    )
    crossed_feature = feature_column.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

    model = define_model(feature_columns=feature_columns)
    model.fit(train_ds, validation_data=val_ds, epochs=5)
    loss, accuracy = model.evaluate(test_ds)
    print(f"Accuracy: {accuracy}, loss: {loss}")

    save_model(model, 0)


if __name__ == "__main__":
    main()
