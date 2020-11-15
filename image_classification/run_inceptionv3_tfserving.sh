#!/bin/sh

set -eu

docker run -t -d --rm \
-p 8501:8501 \
-p 8500:8500 \
--name inception_v3 \
-v $(pwd)/saved_model/inception_v3:/models/inception_v3 \
-e MODEL_NAME=inception_v3 \
tensorflow/serving:2.3.0
