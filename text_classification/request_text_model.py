import argparse
import base64
import json
from re import A
import numpy as np
from typing import Dict, Any
import requests
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def read_text(text_file: str = "./text.txt") -> str:
    with open(text_file, "r") as f:
        text = f.read()
    return text


def request_grpc(
    text: str,
    model_spec_name: str = "text",
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
    request.inputs["text"].CopyFrom(tf.make_tensor_proto([text]))
    response = stub.Predict(request, timeout_second)

    prediction = response.outputs["output_0"].string_val[0].decode("utf-8")
    return prediction


def request_rest(
    text: str,
    model_spec_name: str = "text",
    signature_name: str = "serving_default",
    address: str = "localhost",
    port: int = 8500,
    timeout_second: int = 5,
):
    serving_address = f"http://{address}:{port}/v1/models/{model_spec_name}:predict"
    headers = {"Content-Type": "application/json"}
    request_dict = {"inputs": {"text": [text]}}
    response = requests.post(
        serving_address,
        json.dumps(request_dict),
        headers=headers,
    )
    return dict(response.json())["outputs"][0]


def main():
    parser = argparse.ArgumentParser(description="request inception v3")
    parser.add_argument(
        "-f", "--format", default="GRPC", type=str, help="GRPC or REST request"
    )
    parser.add_argument(
        "-i",
        "--text_file",
        default="./text.txt",
        type=str,
        help="input text file path",
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
        default="text",
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

    text = read_text(text_file=args.text_file)

    if args.format.upper() == "GRPC":
        prediction = request_grpc(
            text=text,
            model_spec_name=args.model_spec_name,
            signature_name=args.signature_name,
            address=args.target,
            port=8500,
            timeout_second=args.timeout_second,
        )
    elif args.format.upper() == "REST":
        prediction = request_rest(
            text=text,
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
