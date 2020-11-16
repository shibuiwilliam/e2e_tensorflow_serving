import argparse
import base64
import json
import numpy as np

import requests
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def read_image(image_file: str = "./a.jpg") -> bytes:
    with open(image_file, "rb") as f:
        raw_image = f.read()
    return raw_image


def request_grpc(
    image: bytes,
    model_spec_name: str = "inception_v3",
    signature_name: str = "serving_default",
    address: str = "localhost",
    port: int = 8500,
    timeout_second: int = 5,
) -> str:
    serving_address = f"{address}:{port}"
    channel = grpc.insecure_channel(serving_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    base64_image = base64.urlsafe_b64encode(image)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    request.model_spec.signature_name = signature_name
    request.inputs["image"].CopyFrom(tf.make_tensor_proto([base64_image]))
    response = stub.Predict(request, timeout_second)

    prediction = response.outputs["output_0"].string_val[0].decode("utf-8")
    return prediction


def request_rest(
    image: bytes,
    model_spec_name: str = "inception_v3",
    signature_name: str = "serving_default",
    address: str = "localhost",
    port: int = 8501,
    timeout_second: int = 5,
):
    serving_address = f"http://{address}:{port}/v1/models/{model_spec_name}:predict"
    headers = {"Content-Type": "application/json"}
    base64_image = base64.urlsafe_b64encode(image).decode("ascii")
    request_dict = {"inputs": {"image": [base64_image]}}
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
        "-i", "--image_file", default="./a.jpg", type=str, help="input image file path"
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
        default="inception_v3",
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

    raw_image = read_image(image_file=args.image_file)

    if args.format.upper() == "GRPC":
        prediction = request_grpc(
            image=raw_image,
            model_spec_name=args.model_spec_name,
            signature_name=args.signature_name,
            address=args.target,
            port=8500,
            timeout_second=args.timeout_second,
        )
    elif args.format.upper() == "REST":
        prediction = request_rest(
            image=raw_image,
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
