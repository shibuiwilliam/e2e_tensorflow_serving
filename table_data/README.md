# E2E TF Serving example for table data
This is an example implementation of an table data classification in TF Serving supporting end-to-end processing from data load to prediction in label.

## Requirements
- Python >= 3.8.5
- Tensorflow >= 2.3.1

## References

- [Classify structured data with feature columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns)

## Why

This repository supports end-to-end processing, from data load, preprocess, prediction, to labelling, of table data classification in TF Serving.
While usual model serving consists of data load, preprocess and labelling in Python, and the TF Serving taking only prediction portion, I tried to cover the procedures required to run the machine learning cycle.
The Tensorflow operators support various data processings including embedding, hashing, bucketization and so on, which can be integrated to be exported to a saved model to be run in TF Serving.
See below example for covering end-to-end table data analysis process in a Tensor operation.

```python
# defining feature column layers
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

# load feature column process as a tf.keras layer
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
```

## How to use

1. Build saved model

```sh
$ python build_model.py
```

2. Run TF Serving docker container

```sh
$ ./run_table_data_tfserving.sh
```

3. See TF Serving metadata

```sh
$ curl localhost:8501/v1/models/table_data/versions/0/metadata
{
"model_spec":{
 "name": "table_data",
 "signature_name": "",
 "version": "0"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": {
   "inputs": {
    "oldpeak": {
     "dtype": "DT_DOUBLE",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_oldpeak:0"
    },
    "restecg": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_restecg:0"
    },
    "trestbps": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_trestbps:0"
    },
    "slope": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_slope:0"
    },
    "sex": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_sex:0"
    },
    "ca": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_ca:0"
    },
    "exang": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_exang:0"
    },
    "fbs": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_fbs:0"
    },
    "chol": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_chol:0"
    },
    "thalach": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_thalach:0"
    },
    "thal": {
     "dtype": "DT_STRING",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_thal:0"
    },
    "cp": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_cp:0"
    },
    "age": {
     "dtype": "DT_INT64",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_age:0"
    }
   },
   "outputs": {
    "output_1": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "StatefulPartitionedCall_1:0"
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
$ python request_table_data.py -f GRPC
0.026179581880569458

# REST API
$ python request_table_data.py -f REST
0.0261795819
```