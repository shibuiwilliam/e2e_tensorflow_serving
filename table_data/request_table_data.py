import argparse
import base64
import json
import numpy as np
from typing import Dict, Any

import requests
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def read_data(data_file: str = "./data.json") -> Dict[str, Any]:
    with open(data_file, "r") as f:
        data = json.load(f)
    return data


def request_grpc(
    data: Dict[str, Any],
    model_spec_name: str = "inception_v3",
    signature_name: str = "serving_default",
    address: str = "localhost",
    port: int = 8500,
    timeout_second: int = 5,
) -> str:
    serving_address = f"{address}:{port}"
    channel = grpc.insecure_channel(serving_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    request.model_spec.signature_name = signature_name

    age = np.array(data["age"], dtype=np.int64)
    sex = np.array(data["sex"], dtype=np.int64)
    cp = np.array(data["cp"], dtype=np.int64)
    trestbps = np.array(data["trestbps"], dtype=np.int64)
    chol = np.array(data["chol"], dtype=np.int64)
    fbs = np.array(data["fbs"], dtype=np.int64)
    restecg = np.array(data["restecg"], dtype=np.int64)
    thalach = np.array(data["thalach"], dtype=np.int64)
    exang = np.array(data["exang"], dtype=np.int64)
    oldpeak = np.array(data["oldpeak"], dtype=np.float64)
    slope = np.array(data["slope"], dtype=np.int64)
    ca = np.array(data["ca"], dtype=np.int64)
    thal = np.array(data["thal"], dtype=str)

    request.inputs["age"].CopyFrom(tf.make_tensor_proto(age))
    request.inputs["sex"].CopyFrom(tf.make_tensor_proto(sex))
    request.inputs["cp"].CopyFrom(tf.make_tensor_proto(cp))
    request.inputs["trestbps"].CopyFrom(tf.make_tensor_proto(trestbps))
    request.inputs["chol"].CopyFrom(tf.make_tensor_proto(chol))
    request.inputs["fbs"].CopyFrom(tf.make_tensor_proto(fbs))
    request.inputs["restecg"].CopyFrom(tf.make_tensor_proto(restecg))
    request.inputs["thalach"].CopyFrom(tf.make_tensor_proto(thalach))
    request.inputs["exang"].CopyFrom(tf.make_tensor_proto(exang))
    request.inputs["oldpeak"].CopyFrom(tf.make_tensor_proto(oldpeak))
    request.inputs["slope"].CopyFrom(tf.make_tensor_proto(slope))
    request.inputs["ca"].CopyFrom(tf.make_tensor_proto(ca))
    request.inputs["thal"].CopyFrom(tf.make_tensor_proto(thal))
    response = stub.Predict(request, timeout_second)

    prediction = response.outputs["output_1"].float_val[0]
    return prediction


def request_rest(
    data: Dict[str, Any],
    model_spec_name: str = "table_data",
    signature_name: str = "serving_default",
    address: str = "localhost",
    port: int = 8500,
    timeout_second: int = 5,
):
    serving_address = f"http://{address}:{port}/v1/models/{model_spec_name}:predict"
    headers = {"Content-Type": "application/json"}
    request_dict = {"inputs": data}
    response = requests.post(
        serving_address,
        json.dumps(request_dict),
        headers=headers,
    )
    return dict(response.json())["outputs"][0][0]


def main():
    parser = argparse.ArgumentParser(description="request inception v3")
    parser.add_argument(
        "-f", "--format", default="GRPC", type=str, help="GRPC or REST request"
    )
    parser.add_argument(
        "-d",
        "--data_file",
        default="./data.json",
        type=str,
        help="input json file path",
    )
    parser.add_argument(
        "-t", "--target", default="localhost", type=str, help="target address"
    )
    parser.add_argument(
        "-s", "--timeout_second", default=5, type=int, help="timeout in second"
    )
    parser.add_argument(
        "-m",
        "--model_spec_name",
        default="table_data",
        type=str,
        help="model spec name",
    )
    parser.add_argument(
        "-n",
        "--signature_name",
        default="serving_default",
        type=str,
        help="model signature name",
    )
    args = parser.parse_args()

    data_json = read_data(data_file=args.data_file)

    if args.format.upper() == "GRPC":
        prediction = request_grpc(
            data=data_json,
            model_spec_name=args.model_spec_name,
            signature_name=args.signature_name,
            address=args.target,
            port=8500,
            timeout_second=args.timeout_second,
        )
    elif args.format.upper() == "REST":
        prediction = request_rest(
            data=data_json,
            model_spec_name=args.model_spec_name,
            signature_name=args.signature_name,
            address=args.target,
            port=8501,
            timeout_second=args.timeout_second,
        )
    else:
        raise ValueError("Undefined format; should be GRPC or REST")
    print(prediction)


if __name__ == "__main__":
    main()
