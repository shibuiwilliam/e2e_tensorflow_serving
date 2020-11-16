# E2E TF Serving example for image classification
This is an example implementation of an image classifier in TF Serving supporting end-to-end processing from data load to prediction in label.

## Requirements
- Python >= 3.8.5
- Tensorflow >= 2.3.1

## Why

This repository supports end-to-end processing, from data load, preprocess, prediction, to labelling, of image classification in TF Serving.
While usual model serving consists of data load, preprocess and labelling in Python, and the TF Serving taking only prediction portion, I tried to cover the procedures required to run the machine learning cycle.
The Tensorflow operators support various data processings including raw file decode, image resize and labelling, which can be exported to a saved model to be run in TF Serving.
See below example for covering end-to-end image classification process in a Tensor operation.

```python
@tf.function(
    input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="image")]
)
def serving_fn(self, input_img: str) -> tf.Tensor:
    def _base64_to_array(img):
        img = tf.io.decode_base64(img)
        img = tf.io.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (299, 299))
        img = tf.reshape(img, (299, 299, 3))
        return img

    img = tf.map_fn(_base64_to_array, input_img, dtype=tf.float32)
    predictions = self.model(img)

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
$ python build_inceptionv3.py
```

2. Run TF Serving docker container

```sh
$ ./run_inceptionv3_tfserving.sh
```

3. See TF Serving metadata

```sh
$ curl localhost:8501/v1/models/inception_v3/versions/0/metadata
{
"model_spec":{
 "name": "inception_v3",
 "signature_name": "",
 "version": "0"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": {
   "inputs": {
    "image": {
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
     "name": "serving_default_image:0"
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
$ python request_inceptionv3.py -f GRPC
Siamese cat

# REST API
$ python request_inceptionv3.py -f REST
Siamese cat
```