#!/bin/sh

set -eu

docker run -t -d --rm \
-p 8501:8501 \
-p 8500:8500 \
--name text \
-v $(pwd)/saved_model/text:/models/text \
-e MODEL_NAME=text \
tensorflow/serving:2.3.0
