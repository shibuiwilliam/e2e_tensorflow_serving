#!/bin/sh

set -eu

docker run -t -d --rm \
-p 8501:8501 \
-p 8500:8500 \
--name table_data \
-v $(pwd)/saved_model/table_data:/models/table_data \
-e MODEL_NAME=table_data \
tensorflow/serving:2.3.0
