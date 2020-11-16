# Tensorflow Serving を使い倒す

有望なディープラーニングのライブラリは Tensorflow と PyTorch で勢力が二分されている現状です。それぞれに強み弱みがあり、以下のような特徴があると思います。

- Tensorflow：Tensorflow Serving や Tensorflow Lite のような豊富な推論エンジン、Tensorflow Hub による多様な学習済みモデルの配布、TFX によるパイプライン、Keras の便利な学習 API
- PyTorch：Define by Run による強力な学習、TorchVision による画像処理

研究や学習では PyTorch が圧倒的になっていますが、推論器を動かすとなると Tensorflow のほうが有力な機能を提供していると思います。PyTorch は ONNX で推論することが可能ですが、モバイル向けや End-to-end なパイプラインサポートとなると、Tensorflow Lite や TFX 含めて Tensorflow が便利です。

本ブログでは Tensorflow Serving を用いた推論器とクライアントの作り方を説明します。Tensorflow Serving を動かすだけであれば多様な記事がありますが、本ブログではデータの入力から前処理、推論、後処理、出力まで、End-to-end で Tensorflow でカバーする方法を紹介します。

## 問題意識

ディープラーニングでモデルを学習した後、モデルは saved model や ONNX 形式で出力できても、前処理や出力が学習時の Python コードしかなく、推論へ移行するときに書き直すことになります。

<TODO：絵>

学習も推論も Python で、Python コードをそのまま使い回せるなら良いですが（それでも間違うことが多々ありますが）、本番システムは Java や Golang、Node.js で Python を組み込む基盤や運用がないということがあります。Python 以外の言語で画像やテーブルデータの処理が Python ほど豊富であるとは限りませんし、Python で実行している前処理をそのまま動かすことができるとは限りません。
解決策のひとつは、機械学習の推論プロセスをサポートする推論器を作ることです。推論プロセスのすべてを Tensorflow の saved model に組み込んでしまい、Tensorflow Serving へ生データをリクエストすれば推論結果がレスポンスされる API を作れば、連携するバックエンドは REST クライアントや GRPC クライアントとして Tensorflow Serving にリクエストを送るだけで良くなります。
Tensorflow の Operator はニューラルネットワークだけでなく、画像のデコードやリサイズ、テーブルデータの One Hot 化等、機械学習に必須な処理が可能になっています。従来であれば Python の Pillow や Scikit-learn に依存していた処理が Tensorflow の計算グラフに組み込まれているため、推論のデータ入力から推論結果の出力まで、全工程を Tensorflow Serving でカバーすることができます。

本ブログでは Tensorflow Serving による画像分類、テキストの感情分析、テーブルデータの 2 値分類を使い、Tensorflow Serving の可能性を開いていきたいと思います。

## Tensorflow Serving

Tensorflow Serving は Tensorflow や Keras のモデルを推論器として稼働させるためのシステムです。Tensorflow の saved model を Web API（GRPC と REST API）として稼働させることができます。また単なる Web API だけでなく、[バッチシステム](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/batching/README.md)として動かすこともできます。[複数バージョンのモデルを同一の Serving に組み込み、エンドポイントを分けることも可能です](https://www.tensorflow.org/tfx/serving/serving_config)。Tensorflow Serving は Docker で起動させることが一般的です。

![Tensorflow Serving](https://www.tensorflow.org/tfx/serving/images/serving_architecture.svg)

## 画像分類

ディープラーニングの重要な使い途の一つが画像処理です。今回は[Inception V3](https://tfhub.dev/google/imagenet/inception_v3/classification/4)を使った画像分類を Tensorflow Serving で動かします。
画像分類のプロセスは以下になります。

1. 生データの画像ファイルを入力データとして受け取る。
2. 画像をデコードする。
3. 画像をリサイズして Inception V3 の入力 Shape である(299,299,3)に変換する。
4. Inception V3 で推論し、Softmax を得る。
5. 各ラベルに Softmax の確率をマッピングする。
6. 最も確率の高いラベルを出力する。

Inception V3 が担うのは常勤お 4 のみで、1,2,3,5,6 は前処理や後処理として周辺システムでカバーする必要があります。学習時は Python で Pillow や OpenCV、Numpy 等々を使って書きますが、推論時に同様のライブラリを使えるとは限りません。特に Python 以外の言語で構築する場合、OpenCV を使うことはできるかもしれませんが、他の Pillow や Numpy は他のライブラリで代替するか、自作する必要があります。
しかし Tensorflow であれば、1,2,3,5,6 も Tensor Operation に組み込み、推論の全行程をカバーすることができます。そのためには tf.function に前処理（1,2,3）と後処理（5,6）の Operation を記述します。
以下の`def serfing_fn`がその Operation になります。Pillow や Numpy でも同様の処理を書くことがあると思いますが、記述量も複雑さも大差ない実装が可能です。

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

上記`InceptionV3Model`クラスのインスタンスを saved model として保存し、Tensorflow Serving として起動することができます。起動した Tensorflow Serving は GRPC として 8500 ポート、REST API として 8501 ポートが開放されます。

```sh
docker run -t -d --rm \
-p 8501:8501 \
-p 8500:8500 \
--name inception_v3 \
-v $(pwd)/saved_model/inception_v3:/models/inception_v3 \
-e MODEL_NAME=inception_v3 \
tensorflow/serving:2.3.0
```

エンドポイントの定義は以下のようになっています。`inputs`以下が入力定義で、`outputs`以下が出力定義です。`inputs`では`image`タグのデータを取ります。Shape が`-1`となっていますが、これは画像の base64 エンコードされたデータを入力とするためです。この時点で Tensorflow Serving への入力は(299,299,3)次元の配列ではなく、画像データそのものとなっています。

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

リクエストは以下のように実行することができます。GRPC と REST API の例を書いていますが、どちらも画像をバイナリデータとして読み込み、base64 エンコードして Tensorflow Serving のエンドポイントにリクエストします。クライアントは前処理することなく Tensorflow Serving にデータをリクエストします。
注意点は Tensorflow の[`tf.io.decode_base64`](https://www.tensorflow.org/api_docs/python/tf/io/decode_base64)が`base64.urlsafe_b64encode`されたデータでないとデコードできないという点です。

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

続いてテキスト分類です。テキスト処理も画像と同様で、入力、前処理、後処理、出力になる箇所を
Tensorflow でカバーします。Tensorflow のテキスト処理で使えるライブラリは複数あります。

- [Tensorflow Text](https://github.com/tensorflow/text)
- [Tensorflow Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)
- [Tensorflow Keras Preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing)
- [Tensorflow Keras Layers Preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing)

今回は[Tensorflow Keras Layers Preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing)を使います。これを選んだのは API が使いやすいという理由です。

データとして[Kaggle にある感情分析の NLP データ](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp)を使用します。感情分析の英文データで、[anger, fear, joy, love, sadness, surprise]の 6 クラス分類となっています。

- anger: i felt anger when at the end of a telephone call
- fear: i pay attention it deepens into a feeling of being invaded and helpless
- joy: i am feeling totally relaxed and comfy
- love: i want each of you to feel my gentle embrace
- sadness: i realized my mistake and i m really feeling terrible and thinking that i shouldn't do that
- surprise: i feel shocked and sad at the fact that there are so many sick people

テキストは前処理として tfidf でベクター化します。[Tensorflow Keras Layers Preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing)では[TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization)で tfidf 等のベクター化が可能です。

## テーブルデータ 2 値分類
