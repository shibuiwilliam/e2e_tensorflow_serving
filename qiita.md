# Tensorflow Servingを使い倒す

有望なディープラーニングのライブラリはTensorflow とPyTorchで勢力が二分されている現状です。それぞれに強み弱みがあり、以下のような特徴があると思います。

- Tensorflow：Tensorflow ServingやTensorflow Liteのような豊富な推論エンジン、Tensorflow Hubによる多様な学習済みモデルの配布、TFXによるパイプライン、Kerasの便利な学習API
- PyTorch：Define by Runによる強力な学習、TorchVisionによる画像処理

研究や学習ではPyTorchが圧倒的になっていますが、推論器を動かすとなるとTensorflowのほうが有力な機能を提供していると思います。PyTorchはONNXで推論することが可能ですが、モバイル向けやEnd-to-endなパイプラインサポートとなると、Tensorflow LiteやTFX含めてTensorflowが便利です。

本ブログではTensorflow Servingを用いた推論器とクライアントの作り方を説明します。Tensorflow Servingを動かすだけであれば多様な記事がありますが、本ブログではデータの入力から前処理、推論、後処理、出力まで、End-to-endでTensorflowでカバーする方法を紹介します。

## 問題意識
ディープラーニングでモデルを学習した後、モデルはsaved modelやONNX形式で出力できても、前処理や出力が学習時のPythonコードしかなく、推論へ移行するときに書き直すことになります。

<TODO：絵>

学習も推論もPythonで、Pythonコードをそのまま使い回せるなら良いですが（それでも間違うことが多々ありますが）、本番システムはJavaやGolang、Node.jsでPythonを組み込む基盤や運用がないということがあります。Python以外の言語で画像やテーブルデータの処理がPythonほど豊富であるとは限りませんし、Pythonで実行している前処理をそのまま動かすことができるとは限りません。
解決策のひとつは、機械学習の推論プロセスをサポートする推論器を作ることです。推論プロセスのすべてをTensorflowのsaved modelに組み込んでしまい、Tensorflow Servingへ生データをリクエストすれば推論結果がレスポンスされるAPIを作れば、連携するバックエンドはRESTクライアントやGRPCクライアントとしてTensorflow Servingにリクエストを送るだけで良くなります。
TensorflowのOperatorはニューラルネットワークだけでなく、画像のデコードやリサイズ、テーブルデータのOne Hot化等、機械学習に必須な処理が可能になっています。従来であればPythonのPillowやScikit-learnに依存していた処理がTensorflowの計算グラフに組み込まれているため、推論のデータ入力から推論結果の出力まで、全工程をTensorflow Servingでカバーすることができます。

本ブログではTensorflow Servingによる画像分類、テキストの感情分析、テーブルデータの2値分類を使い、Tensorflow Servingの可能性を開いていきたいと思います。

## Tensorflow Serving

Tensorflow ServingはTensorflowやKerasのモデルを推論器として稼働させるためのシステムです。Tensorflowのsaved modelをWeb API（GRPCとREST API）として稼働させることができます。また単なるWeb APIだけでなく、[バッチシステム](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md)として動かすこともできます。[複数バージョンのモデルを同一のServingに組み込み、エンドポイントを分けることも可能です](https://www.tensorflow.org/tfx/serving/serving_config)。Tensorflow ServingはDockerで起動させることが一般的です。

![Tensorflow Serving](https://www.tensorflow.org/tfx/serving/images/serving_architecture.svg)

## 画像分類

ディープラーニングの重要な使い途の一つが画像処理です。今回は[Inception V3](https://tfhub.dev/google/imagenet/inception_v3/classification/4)を使った画像分類をTensorflow Servingで動かします。
画像分類のプロセスは以下になります。

1. 生データの画像ファイルを入力データとして受け取る。
2. 画像をデコードする。
3. 画像をリサイズしてInception V3の入力Shapeである(299,299,3)に変換する。
4. Inception V3で推論し、Softmaxを得る。
5. 各ラベルにSoftmaxの確率をマッピングする。
6. 最も確率の高いラベルを出力する。

Inception V3が担うのは常勤お4のみで、1,2,3,5,6は前処理や後処理として周辺システムでカバーする必要があります。学習時はPythonでPillowやOpenCV、Numpy等々を使って書きますが、推論時に同様のライブラリを使えるとは限りません。特にPython以外の言語で構築する場合、OpenCVを使うことはできるかもしれませんが、他のPillowやNumpyは他のライブラリで代替するか、自作する必要があります。
しかしTensorflow であれば、1,2,3,5,6もTensor Operationに組み込み、推論の全行程をカバーすることができます。そのためにはtf.functionに前処理（1,2,3）と後処理（5,6）のOperationを記述します。
以下の`def serfing_fn`がそのOperationになります。PillowやNumpyでも同様の処理を書くことがあると思いますが、記述量も複雑さも大差ない実装が可能です。

```Python
from typing import List
import tensorflow as tf
from tensorflow import keras

class InceptionV3Model(tf.keras.Model):
    def __init__(self, model: tf.keras.Model, labels: List[str]):
        super().__init__(self)
        self.model = model  # Inception V3 model
        self.labels = labels  # ImageNet labels in list

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

    def save(self, export_path="./saved_model/inception_v3/0/"):
        signatures = {"serving_default": self.serving_fn}
        tf.keras.backend.set_learning_phase(0)
        tf.saved_model.save(self, export_path, signatures=signatures)
```

上記`InceptionV3Model`クラスのインスタンスをsaved modelとして保存し、Tensorflow Servingとして起動することができます。起動したTensorflow ServingはGRPCとして8500ポート、REST APIとして8501ポートが開放されます。

```sh
docker run -t -d --rm \
-p 8501:8501 \
-p 8500:8500 \
--name inception_v3 \
-v $(pwd)/saved_model/inception_v3:/models/inception_v3 \
-e MODEL_NAME=inception_v3 \
tensorflow/serving:2.3.0
```

エンドポイントの定義は以下のようになっています。`inputs`以下が入力定義で、`outputs`以下が出力定義です。`inputs`では`image`タグのデータを取ります。Shapeが`-1`となっていますが、これは画像のbase64エンコードされたデータを入力とするためです。この時点でTensorflow Servingへの入力は(299,299,3)次元の配列ではなく、画像データそのものとなっています。

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

リクエストは以下のように実行することができます。GRPCとREST APIの例を書いていますが、どちらも画像をバイナリデータとして読み込み、base64エンコードしてTensorflow Servingのエンドポイントにリクエストします。クライアントは前処理することなくTensorflow Servingにデータをリクエストします。
注意点はTensorflowの[`tf.io.decode_base64`](https://www.tensorflow.org/api_docs/python/tf/io/decode_base64)が`base64.urlsafe_b64encode`されたデータでないとデコードできないという点です。

```python
def read_image(image_file: str = "./a.jpg") -> bytes:
    with open(image_file, "rb") as f:
        raw_image = f.read()
    return raw_image

# GRPC
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

#REST
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
```

推論結果は以下のようになります。

```sh
# GRPC
$ python request_inceptionv3.py -f GRPC
Siamese cat

# REST API
$ python request_inceptionv3.py -f REST
Siamese cat
```

## テキストの感情分析

## テーブルデータ2値分類


