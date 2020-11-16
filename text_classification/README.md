# E2E TF Serving example for text classification
This is an example implementation of an text classification in TF Serving supporting end-to-end processing from data load to prediction in label.

## Requirements
- Python >= 3.8.5
- Tensorflow >= 2.3.1

## References

- [Emotions dataset for NLP](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp)

## Why

This repository supports end-to-end processing, from data load, preprocess, prediction, to labelling, of text classification in TF Serving.
While usual model serving consists of data load, preprocess and labelling in Python, and the TF Serving taking only prediction portion, I tried to cover the procedures required to run the machine learning cycle.
The Tensorflow operators support various data processings including embedding, tfidf tokenization and so on, which can be integrated to be exported to a saved model to be run in TF Serving.
See below example for covering end-to-end text analysis process in a Tensor operation.

```python
# define tokenization layers
text_vectorizer = make_text_vectorizer(data=train_data)

# define model
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = layers.Dense(1)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(6, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# make serving function
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
```

## How to use

1. Build saved model

```sh
$ python build_text_model.py
```

2. Run TF Serving docker container

```sh
$ ./run_text_model_tfserving.sh
```

3. See TF Serving metadata

```sh
$ curl localhost:8501/v1/models/text/versions/0/metadata
{
"model_spec":{
 "name": "text",
 "signature_name": "",
 "version": "0"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": {
   "inputs": {
    "text": {
     "dtype": "DT_STRING",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_text:0"
    }
   },
   "outputs": {
    "output_0": {
     "dtype": "DT_STRING",
     "tensor_shape": {
      "dim": [],
      "unknown_rank": true
     },
     "name": "StatefulPartitionedCall:0"
    }
   },
   "method_name": "tensorflow/serving/predict"
  },
  "__saved_model_init_op": {
   "inputs": {},
   "outputs": {
    "__saved_model_init_op": {
     "dtype": "DT_INVALID",
     "tensor_shape": {
      "dim": [],
      "unknown_rank": true
     },
     "name": "NoOp"
    }
   },
   "method_name": ""
  }
 }
}
}
}
```

4. Send request

```sh
# GRPC
$ python request_text_model.py -f GRPC
sadness

# REST API
$ python request_text_model.py -f REST
sadness
```