[[Japanese](README.md)/[English](README_EN.md)]

# Intern-Decision-0.8B-ONNX-CPU

[`internlm/Intern-Decision-0.8B`](https://huggingface.co/internlm/Intern-Decision-0.8B)（構造化判定モデル。テキストと画像）を、onnxruntime と numpy だけで推論する最小依存の推論実装です。 ※GPU推論も可<br>
公式の `inference.py` が要求する torch・torchvision・transformers は使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ（画像を使う場合は pillow を足した3つ）
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ。画像は `demo_inference_image.py`、カメラは `demo_inference_webcam.py`
- 判定は参照実装（PyTorch fp32）と一致（テキスト 10/10、画像 4/4）
- 全質問が1回の順伝播で決まる（生成は走らない）。問いを増やしてもプロンプトが伸びるだけで、8問は1問より約2%遅いだけ
- 画像対応（1回に最大8枚。画像は1枚ずつエンコードして連結）
- 前処理（Qwen byte-level BPE、チャットテンプレート、3次元 M-RoPE、画像のパッチ化）を純Pythonで移植し、参照実装との突き合わせ検証を同梱
- CPU のほか、GPU（CUDA または DirectML）でも動かせる（`--gpu`）

# Purpose of This Repository
以下の検証を目的としています。
- Intern-Decision の判定を、実行時依存2〜3つだけでどこまで動かせるか
- 参照実装の前処理を純Pythonに移植して、同じ入力が作れるか
- CPU（と GPU）で速度・メモリがどうなるか

実測値と移植の根拠は[artifacts/report.md](artifacts/report.md)を参照してください。

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   （画像を使う場合のみ）
```
実行に必要なのはこれだけです。モデルの取得も標準ライブラリのみで動きます。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
オプションのグループがあります。
- `verify`：トークナイザの突き合わせ検証（`check_tokenizer_fuzz`）用の `tokenizers`
- `webcam`：`demo_inference_webcam.py` 用の OpenCV
- `export`：ONNX を自分で書き出すときの torch・transformers など（実行時には使いません）

# Installation

```bash
git clone https://github.com/<OWNER>/Intern-Decision-0.8B-ONNX-CPU
cd Intern-Decision-0.8B-ONNX-CPU

# 実行時の依存（onnxruntime・numpy・pillow）のみインストール
uv sync
```

# Download Model
Intern-Decision には公開されている ONNX がありません。書き出した ONNX を[Releases](https://github.com/<OWNER>/Intern-Decision-0.8B-ONNX-CPU/releases)に置いています（リポジトリには含めていません）。<br>
以下のスクリプトで `models/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py --repo <OWNER>/Intern-Decision-0.8B-ONNX-CPU
```
- 既定の系列長 L=384 の分（約 3.4 GB）を取得し、Release の `manifest.json` に書かれた SHA-256 と照合します（不一致ならファイルを削除して停止）
- GitHub の制限（1ファイル 2 GiB 未満）で分割されたファイルは、取得後に結合して照合します
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します
- `INTERN_ONNX_REPO=<OWNER>/Intern-Decision-0.8B-ONNX-CPU` を環境変数に入れておけば、`--repo` は不要です

```bash
uv run download_model.py --repo <OWNER>/... --list                       # ファイル一覧
uv run download_model.py --repo <OWNER>/... --exclude "vision_encoder*"  # テキスト専用（視覚エンコーダを取らない。約 0.4 GB 少ない）
uv run download_model.py --repo <OWNER>/... --tag v1                     # タグを指定
```

Release には L=384（既定）・512・1024・1280 の4つのグラフが入っています（全部で約 4.3 GB）。**重み（`weights.onnx.data*`、約 3.0 GB）は4つで共有しているので、グラフを1つ増やしても約 0.25 GB しか増えません。** 既定の L=384 だけ取るには、次のどちらかを使います（どちらも同じ 14 ファイル、約 3.4 GB になります）。
```bash
uv run download_model.py --repo <OWNER>/... --seq-len 384
uv run download_model.py --repo <OWNER>/... --exclude "model_L512*" --exclude "model_L1*"
# glob で「含める」形で書くなら、共有ファイルも並べます
uv run download_model.py --repo <OWNER>/... --include "*L384*" --include "weights*" --include "vision*" --include "*.json" --include "hf/*"
```
あとで L=512 だけ足すときは `--seq-len 512`（取得済みの重みは再取得しません）。

自分で書き出す場合（元の checkpoint を取得して ONNX にします。torch が要ります）。
```bash
uv sync --group export
uv run export/prepare_model.py                  # 既定: L=384
uv run export/prepare_model.py --seq-len 512    # 2つ目以降のバケット（重みは共有され、グラフだけ増えます）
uv run export/prepare_model.py --seq-len 1024   # 問いを増やしたいとき
```

# Usage

### 実行例(Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
画像のデモは、既定で `artifacts/images/sample_chart.png`（棒グラフ。「どの棒が最も高いか」「棒グラフか」を聞きます）を使います。このサンプルは、Wikipedia 画像ベンチの合成データ（スクリプトで描いた画像。`syn-0001`）の1枚で、写真ではなく、Wikipedia の素材も含みません。自分の画像を使うときは `--image` を指定します。
```bash
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```
**画像は、モデルに渡す前に、長辺 224 px に縮小します**（`--size`、既定 224。`0` で縮小しません）。グラフは L トークンで固定です（既定 L=384）。448 px の画像は約 570〜600 トークンになり L=384 には収まらず（L=1024 のグラフがあればそこへ自動で切り替わります）、224 px なら約 64 トークンで済みます。

```python
from intern_decide import DecisionEngine

engine = DecisionEngine("models/onnx", tokenizer_json="models/hf/tokenizer.json")   # 既定: L=384（seq_len=512 などで変えられます）

engine.predict({
    "state": "I was charged twice for my Pro subscription this month.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "charges, refunds", "technical": "bugs"}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["low", "medium", "high", "critical"]},
    },
})
```

**全フィールドが1回の順伝播で決まります。** assistant ターンが `<decision>` を値に持つ JSON スケルトンで、各マーカーの**1つ前**の logits がその項目の答えを決めます。生成は一切走りません。

画像は同じリクエストに混ぜます（最大8枚）。
```python
engine.predict({"state": "Look at the screenshot.", "images": ["shot.png"],
                "questions": {...}})
```
`images` には、ファイルのパス、`data:image/...;base64,` の URL、バイト列、Pillow の画像を渡せます。

### カメラで「人が映っているか」を判定する
OpenCV でカメラの映像を取得し、最新のフレームについて「人が映っているか」を判定して、結果を映像に重ねて表示します。
```bash
uv run --group webcam demo_inference_webcam.py                        # カメラ 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/sample_chart.png   # カメラの代わりに画像・動画ファイル（人は映っていないので NO になります）
```
- 1回の判定は CPU で数秒かかるため、判定は別スレッドで動かし、映像の表示は止めません。判定が出るたびに、上部の帯（緑＝はい／赤＝いいえ）と「はい」の確率が更新されます
- 質問は `--question`（はい・いいえの質問）で変えられます。モデルに渡す画像は長辺 224 px に縮小します（`--size`）
- `q` または `Esc` で終了します。OpenCV は、このデモだけの依存です

### 質問の種類
| `type` | 答え |
|---|---|
| `choice` | 選択肢から1つ（`choice`、`confidence`、`probabilities`） |
| `score` | 順序尺度の期待値（`score`、`legend`）。先頭が最低 |
| `noul` | 文が成り立つか（`noul` = はいの確率）。選択肢は `yes` / `no` 固定 |

### GPUで実行する（任意）
`--gpu` を付けると、CUDA（または DirectML）で推論します。この onnxruntime ビルドに GPU プロバイダが無い場合は、警告を出して CPU で動きます。GPU に載らない場合も、警告を出して CPU で動きます。
```bash
uv run --no-sync demo_inference_text.py --gpu
uv run --no-sync verify/bench.py --gpu
```
```python
from intern_decide import DecisionEngine, gpu_providers

engine = DecisionEngine("models/onnx", tokenizer_json="models/hf/tokenizer.json", providers=gpu_providers())
print(engine.provider)   # 実際に使われたプロバイダ
```

注意点です。
- `onnxruntime` の代わりに `onnxruntime-gpu`（CUDA。CUDA 13 と cuDNN 9 が PATH に必要）または `onnxruntime-directml`（Windows。CUDA の導入は不要）を入れます。これらは同じ `onnxruntime/` ディレクトリにファイルを書くため、併存させないでください。入れ替えたときは `uv pip install --force-reinstall --no-deps onnxruntime-directml` で入れ直します
- **CUDA を使うときは、CUDA 13 と cuDNN 9 の `bin` フォルダを PATH に入れてください。** `onnxruntime-gpu` は CUDA の DLL を同梱せず、PATH から探します。インストールしただけでは足りない場合があります（`CUDA_PATH` が古い CUDA を指していることがあります）。見つからないと `cublasLt64_13.dll` が無いというエラーが出て、CPU で動きます。Windows の例です（バージョンは環境に合わせてください）。
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
  PATH を変えたあとは、ターミナルを開き直します
- `uv run` は `pyproject.toml` に合わせて `onnxruntime`（CPU版）を再インストールするため、`uv run --no-sync` を使ってください（`UV_NO_SYNC=1` でも同じです）
- 実際に使われたプロバイダは、デモの最終行と `verify/bench.py` の出力（`provider`）で確認できます
- fp32 のグラフは重みだけで約 3.3 GB あります。VRAM が小さい GPU では載らず、CPU に切り替わります。測定結果は「性能」の節を参照してください
- GPU での確率は、CPU と完全には一致しません（演算の順序の違い）。参照実装との一致検証は CPU のみです

# 系列長は固定です（重要）
モデルの linear attention が `for i in range(num_chunks)` という Python ループで64トークンのチャンクを回すため、**エクスポート時に展開されて固定回数になります**。動的な系列長のグラフは作れません。

そのため固定長 L で書き出し、推論時に L まで**右パディング**します。読み出し位置はすべてパディングより前で、モデルは causal なので結果は変わりません（実測 1.9e-05）。**コストは実トークン数ではなく L に比例します。**

L=384、512、1024、1280 の4つを書き出してあり、**既定は L=384** です（`DecisionEngine(..., seq_len=384)`、デモと `verify/bench.py` は `--seq-len`）。224 px の画像1枚＋1問のプロンプトは 260〜375 トークンで、L=384 なら 100 % 収まります（L=256 は 20 % 未満、L=320 は 94〜99 %）。評価用の 2x1875 プロンプト（224 px）も L=384 に 100 % 収まります。

- `seq_len` で指定した `model_L{seq_len}.onnx` を読みます。無ければ、あるものの中で次に大きいバケットを読みます
- リクエストが収まらないときは、**収まる最小のバケット**に自動で切り替えます（古いグラフを解放してから読むので、大きなグラフがメモリに2つ載ることはありません。視覚エンコーダは載せたままです）。切り替えたあとは、そのバケットのままです
- 収まるバケットが1つも無いときだけ、必要な L を示す `ValueError` になります
- `graph="model_L512.onnx"` を明示したときはそれが優先され、切り替えは起きません（`graph_meta.json` も従来どおり使えます）
- 4つのグラフは**重みのファイルを共有**します（`models/onnx/onnx/weights.onnx.data*`。バイト列が同一のテンソルを同じ位置から読むだけなので、数値は変わりません）

| L | 収まる問い数 | 1回（テキスト2問、ウォーム、5回の中央値） | セッションロード | ピークRAM |
|---|---|---:|---:|---:|
| **384（既定）** | 〜4問（224 px の画像1枚＋1問も収まる） | **2.23 s** | 32 s | 5.4 GB |
| 512 | 〜6問 | 3.07 s | 32 s | 6.1 GB |
| 1024 | 〜13問 | 6.13 s | 38 s | 10.6 GB |
| 1280 | 16問（プロトコル上限） | 7.66 s | 36 s | 13.8 GB |

測定は Core i7-12800H、CPU 6スレッド、fp32、`uv run --no-sync verify/bench.py --seq-len N --questions 2 --runs 5`（253〜256 トークン）です。重いジョブ同士は排他していますが（`heavy.lock`）、軽い作業は同じマシンで動いていた可能性があります。問い数は、サンプルの状態文（約110トークン）に選択式の問いを足したときの目安です。

# Verification
```bash
uv run verify/run_all.py                                 # 実行時依存だけで通るゲート
uv run --group verify verify/run_all.py --full-fuzz      # tokenizers があればファズも本気で回す
```

| ゲート | 内容 | 必要なもの |
|---|---|---|
| check_prompt | プロンプト文字列・`input_ids`・読み出し位置の完全一致 | 実行時依存のみ |
| check_buckets | バケットの選択、収まらないときの切り替え、どれも収まらないときの `ValueError`、`graph=` の優先 | 実行時依存のみ |
| verify | 判定と確率の一致（10/10）。`--seq-len N` で各バケットを検証 | 実行時依存のみ |
| check_image | 前処理が processor の `pixel_values` と一致 | + pillow |
| verify_images | 画像判定の一致（4/4） | + pillow |
| check_tokenizer_fuzz | Rust `tokenizers` との突合 | + tokenizers |
| check_padding | 右パディングが読み出し logits を動かさないこと | + torch |
| probe_lengths | 固定長でなければならない理由の実測 | + torch |

`artifacts/golden.json` と `golden_images.json` は、参照実装（PyTorch fp32、CPU）を実際に動かして作った正解データです。同じ重みを両者が走らせるので、差が出たら移植のバグと断定できます。

### 性能（Core i7-12800H、CPUのみ、fp32、L=512。バケットごとの値は「系列長は固定です」の表）
| 項目 | 値 |
|---|---:|
| テキスト2問 | 3051 ms |
| 画像1枚 + 2問 | 3296 ms |
| セッションロード | 32 s |
| ピークRAM | 5.8 GB |

スレッド数は8が最良です。

### GPU での測定（NVIDIA GeForce RTX 3050 Ti Laptop GPU、VRAM 4 GB、`verify/bench.py`、テキスト2問・253トークン）
| 実行 | 中央値 | セッションロード | 備考 |
|---|---:|---:|---|
| CPU（6スレッド） | 2990 ms | 37 s | ピークRAM 6.1 GB |
| GPU（CUDA、`onnxruntime-gpu` 1.31、CUDA 13.4、cuDNN 9.20） | 3619 ms | 48 s | VRAM 3867 / 4096 MiB を使用、GPU 使用率 97〜98% |
| GPU（DirectML、`onnxruntime-directml` 1.24） | — | — | 初期化に失敗し、警告を出して CPU で動いた |

- **この GPU では、CPU より速くなりません。** fp32 のグラフは重みだけで約 3.3 GB あり、4 GB の VRAM がほぼ満杯になります。ウォームアップの影響ではありません。同じ入力を16回測ると、2回目以降は 3.1〜3.25 秒で一定でした。1回の順伝播で約2万2600個の小さな演算（linear attention を展開したもの）が走り、GPU 上の演算時間の合計は約0.57秒ですが、実時間は約3.1秒です。演算を1つずつ起動する側の待ちが支配的で、GPU の計算力が活きません（CPU で動いたノードは1%程度）。VRAM の大きい GPU での測定は、していません
- 初回の測定は 8427 ms でしたが、ほかの重い処理と重なっていました。測り直した値が上の 3619 ms です
- DirectML では、`'utf-8' codec can't decode` というエラーで初期化に失敗しました（原因は未調査です。メッセージは文字コードの変換エラーです）。コードは警告を出して CPU に切り替えます

# Limitations
- 系列長は固定です。既定は L=384 で、収まらなければ大きいバケットに切り替えます。最大の L を超えると例外になります（必要な `--seq-len` を提示します）
- fp32 のみです。動画は未対応です
- `tokenizer.py` / `chat.py` はこのチェックポイント専用で、想定外の入力は例外で弾きます
- エクスポート時に `optimize=True` は使わないでください（24層で70分以上終わりません）
- GPU では確率が CPU と完全には一致しません。VRAM が小さいと載らず CPU に切り替わります

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は onnxruntime・numpy・pillow。グループ: verify / webcam / export）
download_model.py        # モデル取得（GitHub Releases、SHA-256照合、分割ファイルの結合）
demo_inference_text.py   # テキストの推論デモ
demo_inference_image.py  # 画像の推論デモ
demo_inference_webcam.py # カメラで「人が映っているか」を判定（OpenCV）
intern_decide/           # 推論実装（軽い依存のみ。export/ を import しない）
  runtime.py             #   ORT セッション + 右パディング + 読み出し + GPU プロバイダ
  tokenizer.py           #   Qwen byte-level BPE の純Python実装
  protocol.py            #   inference.py のスキーマ・温度スケーリング・argmax を移植
  chat.py                #   チャットテンプレート（jinja2 なし）と3次元 M-RoPE
  image.py               #   Pillow デコード + smart_resize + パッチ化
  _uniprops.py           #   生成物: Unicode範囲表
export/                  # 重い依存（torch / transformers / onnx）
  prepare_model.py       #   checkpoint 取得 + ONNX 書き出し
  export_onnx.py         #   単一グラフ（読み出し gather 内蔵）+ vision encoder
  share_external_data.py #   バケット間で重みのファイルを共有（同一バイト列のテンソルだけ）
  make_golden.py         #   artifacts/golden.json を作る
  vision_export_patch.py #   単一画像版への差し替え
release/
  make_release.py        # models/ から Release 用のファイルを作る（分割・SHA-256・manifest.json）
verify/                  # 検証ゲート（run_all.pyで一括）
artifacts/               # golden.json（検証の正解データ）、report.md
models/                  # ONNX・トークナイザ・設定（gitignore）
```

# Reference
- [internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B)（元のモデル）

# License
Intern-Decision-0.8B-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B)、Apache-2.0。ベースの Qwen3.5 も Apache-2.0）のライセンスに従います。Release には、両方のライセンス文を同梱しています。
