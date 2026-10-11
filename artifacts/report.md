# Intern-Decision-0.8B の ONNX 化と最小依存推論 — 検証結果

計測: 2026-09-28 / Windows 11 / Intel Core i7-12800H / Python 3.12 (uv) / CPU のみ
対象: [`internlm/Intern-Decision-0.8B`](https://huggingface.co/internlm/Intern-Decision-0.8B)（Qwen3.5-0.8B のファインチューン）

## 結論

**テキストは `onnxruntime` + `numpy` の 2 パッケージ、画像を使う場合も `Pillow` を足した
3 パッケージで動く。** 公式の `inference.py` が要求する torch / torchvision /
transformers は不要になった。

判定結果は参照実装と一致する: **10/10 の判定が同一、確率の最大差 1.7e-06**。

ただし **系列長が固定**という制約が付く（後述）。これはこの実装の都合ではなく、
モデルの linear attention が Python ループでチャンクを回していることに由来する。

## 1. 依存パッケージ

| 構成 | パッケージ数 | サイズ |
|---|---:|---:|
| 公式 `requirements.txt`（torch/torchvision/transformers/Pillow） | 46 | 833 MB |
| **この実装 テキストのみ `.venv-bare`** | **2** | **86 MB** |
| **この実装 画像あり `.venv-min`** | **3** | **100 MB** |

`.venv-bare` で不在を確認済み:

```
torch         ModuleNotFoundError: No module named 'torch'
tokenizers    ModuleNotFoundError: No module named 'tokenizers'
transformers  ModuleNotFoundError: No module named 'transformers'
jinja2        ModuleNotFoundError: No module named 'jinja2'
onnx          ModuleNotFoundError: No module named 'onnx'
regex         ModuleNotFoundError: No module named 'regex'
PIL           ModuleNotFoundError: No module named 'PIL'
```

`jinja2` はチャットテンプレート、`regex` は Qwen の pre-tokenizer が使う
`\p{L}` 等の Unicode クラス、`Pillow` は画像用。いずれも自前実装ないし
テキスト専用化で外した。`onnx` はエクスポート時のみ必要。

## 2. パイプラインの正しさ

### 2.1 プロンプトと読み出し位置（ゲート 1-2）

`inference.py` が実際に組み立てる文字列・トークン列と **5/5 完全一致**:

```
ok  support_routing  prompt=OK ids=OK positions=OK fields=OK symbols=OK  (253 tokens)
ok  incident         prompt=OK ids=OK positions=OK fields=OK symbols=OK  (312 tokens)
ok  bug_triage       prompt=OK ids=OK positions=OK fields=OK symbols=OK  (221 tokens)
ok  many_options     prompt=OK ids=OK positions=OK fields=OK symbols=OK  (271 tokens)
ok  japanese_state   prompt=OK ids=OK positions=OK fields=OK symbols=OK  (244 tokens)
```

移植で気をつけた点:

- 最終ターンが **assistant で、しかも中身（JSON スケルトン）を持つ**。
  `add_generation_prompt=False` かつ最後のユーザー発話より後ろなので、
  テンプレートは空の think ブロックを挟む:
  `<|im_start|>assistant\n<think>\n\n</think>\n\n{skeleton}<|im_end|>\n`
- `noul` の選択肢は **`no` が先**（シンボル A）
- 読み出しは `<decision>` トークンの **1 つ前**の位置
- `argmax` の同点はラベルの辞書順で決まる

### 2.2 トークナイザ（ゲート 3）

Qwen byte-level BPE の純 Python 実装を Rust `tokenizers` と突き合わせ:

```
cases: 187770   mismatches: 0
```

このチェックポイントの `tokenizer.json` は OpenJev-Multimodal のものと別物
（19,989,530 バイト vs 19,226,111、`<decision>`=248077 や `<tts_*>` が追加）なので、
コードは流用しつつ**このファイルに対して**取り直した。
コーパスには 62 個の解答シンボル、`<decision>`、スキーマ行、JSON スケルトンを含めた。

### 2.3 ONNX の数値（ゲート 4-5）

```
decisions 10/10   worst |dp| 1.675e-06
worst |dlogit| vs torch: 2.670e-05   (L=512)
failures: 0 / 5
```

## 3. 系列長が固定である理由

エクスポート直後の版は **253 トークンちょうどで作ったグラフが、221〜256 トークンでは
完全一致するのに 271 / 312 トークンでは静かに誤答**した。128 トークンでは
`Gather` が `idx=3 must be within the inclusive range [-2,1]` で落ちた。

原因は `transformers/models/qwen3_5/modeling_qwen3_5.py` の
`chunk_gated_delta_rule`:

```python
pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
num_chunks = total_sequence_length // chunk_size   # chunk_size = 64
...
for i in range(num_chunks):      # ← Python ループ
```

`num_chunks` はトレース時に Python の int になるため、**このループは展開されて
固定回数になる**。253 トークンでトレースすると 4 チャンク分だけが焼き込まれ、
5 チャンク必要な入力では最後のチャンクが無視され、
2 チャンクしかない入力では範囲外アクセスになる。

`torch.onnx.export` でこれを動的にする方法はない（`torch.while_loop` 等への
書き換えは上流の変更になる）。

**採った対策**: 固定長 L でエクスポートし、**推論時に L まで右パディング**する。
併せて **系列長を静的次元**にしたので、長さが違う入力は黙って壊れず
ORT が明示的に落とす。

### パディングが安全であることの実測

読み出し位置はすべてパディングより前にあり、モデルは causal なので
パディングは届かない。念のため torch で直接確認した:

```
support_routing  253 tokens -> L=384   max|dlogit| = 1.585e-05  OK
support_routing  253 tokens -> L=512   max|dlogit| = 1.526e-05  OK
support_routing  253 tokens -> L=640   max|dlogit| = 1.812e-05  OK
incident         312 tokens -> L=512   max|dlogit| = 1.860e-05  OK
...
worst padding-induced logit change: 1.860e-05
```

float32 の誤差レベルで、パディング量にも依存しない。

## 4. エクスポート

単一グラフ、プレフィル 1 回のみ（`use_cache=False`）なので KV / 再帰状態は
グラフ境界を越えない:

```
input_ids [1, L], attention_mask [1, L], position_ids [3, 1, L], read_positions [K]
    -> logits [1, K, 248320]
```

**読み出し位置の gather はグラフに内蔵した。** 入れないと出力が
`[1, L, 248320]` になり、L=512 で 1 回あたり 500 MB。
読まない位置の語彙射影も無駄になる。

| ファイル | サイズ |
|---|---:|
| `model_L512.onnx` + `.data` | 229 MB + 2870 MB = **3.1 GB** |
| `model_L1024.onnx` + `.data` | 236 MB + 2870 MB = **3.1 GB** |
| 元の safetensors（bf16, 3 分割） | 1627 MB |

fp32 なので元の bf16 の約 2 倍。重み本体（`.data`）は L によらず同一。

> **`optimize=True` は使わないこと。** onnxscript のグラフ最適化に入ると
> 24 層では 70 分経っても終わらなかった（2 層のスモークでは 122 秒）。
> 無効にすると **335 秒（L=512 は 100 秒）**で完了する。

## 5. レイテンシ（CPU）

計測機: Core i7-12800H（14 コア / 20 スレッド）、RAM 32 GB。
GLiNER2.5-Decide を基準にマシンが静かな状態であることを確認済み（throttle x0.9）。

### テキストと画像

| 入力 | トークン | median |
|---|---:|---:|
| テキスト 2 問 | 253 | 3051 ms |
| 画像 1 枚 + 2 問 | 308（うち視覚 70） | **3296 ms** |

画像は **+8 % だけ**。vision encoder が 12 層と小さく、デコーダのコストは
実トークン数ではなく L で決まるため。

### スレッド数（L=512, fp32, 2 問）

| threads | 2 | 4 | **8** | 14 |
|---|---:|---:|---:|---:|
| median ms | 4190 | 3213 | **3051** | 3498 |

8 スレッドが最良。セッションロード **32 秒**、ピーク RAM **5.8 GB**（fp32 のため）。

### 問い数はほぼ無料（ここがこのモデルの要点）

**全フィールドが 1 回の順伝播で決まる。** 問いを増やしてもプロンプトが伸びるだけ:

| L | 1 問 | 4 問 | 8 問 |
|---|---:|---:|---:|
| 512 | 3053 ms | 3118 ms | （512 超） |
| 1024 | 6447 ms | 6463 ms | 6575 ms |

L=1024 で **8 問が 1 問より 2 % 遅いだけ**。1 判定あたり 6447 → 822 ms になる。

### L の選び方

コストは実際のトークン数ではなく **L に比例**する（L=1024 は L=512 の約 2.1 倍）。
**想定リクエストの少し上の L を選ぶ**のが正しい。目安:

| L | 収まる問い数の目安 | 1 回 |
|---|---|---:|
| 512 | 〜6 問 | 3.1 s |
| 1024 | 〜13 問 | 6.4 s |

プロトコルの上限 16 問には L=1280 程度が要る。

## 6. 画像対応

参照実装と同じ結果が出る。テストは色 2 種と図形 1 種、および複数画像:

```
ok  colour_red     red.png     colour ours=red    ref=red    max|dp|=4.40e-07
ok  colour_blue    blue.png    colour ours=blue   ref=blue   max|dp|=1.86e-07
ok  shape_circle   circle.png  shape  ours=circle ref=circle max|dp|=4.77e-07
                               dark   ours=yes    ref=yes    max|dp|=1.36e-07
decisions 4/4   worst |dp| 4.766e-07
```

複数画像（red + blue の 2 枚、374 トークン）も参照と一致（最大差 2.26e-06）。

### 前処理は画素まで完全一致

`prepare_images` の出力を processor が実際に作った `pixel_values` と突き合わせ:

```
ok  colour_red     grid=OK shape=OK tokens=OK prompt=OK ids=OK  max|dp|=0.0000
ok  colour_blue    grid=OK shape=OK tokens=OK prompt=OK ids=OK  max|dp|=0.0000
ok  shape_circle   grid=OK shape=OK tokens=OK prompt=OK ids=OK  max|dp|=0.0000
```

`min_pixels` / `max_pixels` は `preprocessor_config.json` から読む
（65536 / 16777216 = 64 〜 16384 visual トークン）。OpenJev-Multimodal で使った
トークン予算のキャップは既定では掛けない（掛けると参照と答えが変わるため）。

### vision encoder は 1 画像ずつ

`modeling_qwen3_5.py` の 2 箇所が `image_grid_thw` の**値**から
`.tolist()` を呼ぶため、そのままでは `torch.export` が通らない:

- `Qwen3_5VisionAttention.forward` … 画像ごとに attention を区切る `lengths.tolist()`
- `get_image_features` … 出力を画像ごとに割る `split_sizes.tolist()`

どちらも**複数画像を混ぜないため**だけに存在する。1 回 1 画像なら両方とも no-op に
なるので、`scripts/vision_export_patch.py` で単一画像版に差し替えてエクスポートし、
ランタイムは画像を 1 枚ずつエンコードして連結する。
**上の複数画像テストが、この差し替えが参照と同値であることの裏付け**になっている。

| グラフ | サイズ | エクスポート時間 |
|---|---:|---:|
| `vision_encoder.onnx` + `.data` | 3 MB + 402 MB | 21 秒 |

画像特徴はメイングラフの `image_features` / `image_slots` 入力から
トークン埋め込みに散布される（デコーダを 2 つ持たずに済む）。
M-RoPE の 3 次元位置も画像がある場合は `get_rope_index` 相当の計算に切り替わる。

## 7. 実装

`intern_decision/` — 実行時依存は `onnxruntime` / `numpy` のみ。

| ファイル | 役割 |
|---|---|
| `_uniprops.py` | 生成物。`\p{L}` `\p{N}` `\p{M}` `White_Space` の Unicode 範囲表 |
| `bpe.py` | Qwen byte-level BPE の純 Python 実装 |
| `protocol.py` | `inference.py` のスキーマ組み立て・温度スケーリング・argmax を移植 |
| `chat.py` | チャットテンプレート（jinja2 なし）と 3 次元 M-RoPE |
| `runtime.py` | ORT セッション、右パディング、読み出し |
| `demo.py` | CLI |
| `image.py` | 画像前処理（Pillow デコード + `smart_resize` + パッチ化） |

```python
from intern_decision import DecisionEngine

engine = DecisionEngine("weights/onnx", tokenizer_json="weights/hf/tokenizer.json")
engine.predict({
    "state": "I was charged twice for my Pro subscription.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "charges, refunds", "technical": "bugs"}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {}},
    },
})
```

## 8. 制約

- 画像を使うと `Pillow` が要る（デコード用）。テキストだけなら 2 パッケージのまま
- 動画（`<|video_pad|>`）は未対応
- **系列長が固定。** 上限を超えると例外（必要な `--seq-len` を提示する）。
  黙って切り詰めることはしない
- fp32 のみ。RAM 5.8 GB、ロード 32 秒。fp16 なら半減するはずだが未検証
- `bpe.py` / `chat.py` はこのチェックポイント専用。想定外の
  `tokenizer.json` やテンプレート分岐はロード時／実行時に例外で弾く

## 9. 再現手順

```powershell
# 実行時（2 パッケージ）
uv venv .venv-bare --python 3.12
uv pip install --python .venv-bare/Scripts/python.exe --no-deps onnxruntime numpy

.venv-bare\Scripts\python.exe scripts\check_prompt.py     # 5/5 完全一致
.venv-bare\Scripts\python.exe scripts\verify.py           # 10/10, |dp| 1.7e-06
.venv-bare\Scripts\python.exe scripts\bench.py --threads 8
.venv-bare\Scripts\python.exe -c "import torch"           # ModuleNotFoundError

# エクスポートとオラクル（torch / transformers / tokenizers が要る）
uv venv .venv-ref --python 3.12
uv pip install --python .venv-ref/Scripts/python.exe torch torchvision `
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-ref/Scripts/python.exe transformers onnx onnxscript tokenizers

.venv-ref\Scripts\python.exe scripts\make_golden.py               # 参照出力
.venv-ref\Scripts\python.exe scripts\export_onnx.py --seq-len 512 # 100 秒
.venv-ref\Scripts\python.exe scripts\check_padding.py             # パディング安全性
.venv-ref\Scripts\python.exe scripts\fuzz_tokenizer.py 150000     # mismatches: 0
```
