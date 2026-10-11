[[Japanese](README.md)/[English](README_EN.md)]

# Intern-Decision-0.8B-ONNX-CPU

A minimal-dependency inference implementation of [`internlm/Intern-Decision-0.8B`](https://huggingface.co/internlm/Intern-Decision-0.8B) (structured decisions over text and images) on onnxruntime and numpy only. GPU inference is also possible.<br>
It does not use torch, torchvision or transformers, which the official `inference.py` requires.

# Features
- Minimal dependencies: only onnxruntime and numpy at run time (plus pillow when images are used)
- Two scripts are all you need: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference). Images: `demo_inference_image.py`; camera: `demo_inference_webcam.py`
- Decisions match the reference implementation (PyTorch fp32): text 10/10, images 4/4
- Every question is decided in one forward pass (nothing is generated). More questions only lengthen the prompt: 8 questions cost about 2 % more than 1
- Images are supported (up to 8 per request; each is encoded on its own and the results are concatenated)
- The preprocessing (Qwen byte-level BPE, chat template, 3-D M-RoPE, image patching) is ported to pure Python, with gates that compare it against the reference
- Besides the CPU it runs on a GPU (CUDA or DirectML) with `--gpu`

# Purpose of This Repository
- How far Intern-Decision's decisions can be run with only 2-3 runtime dependencies
- Whether the reference implementation's preprocessing can be ported to pure Python and produce the same inputs
- What speed and memory look like on CPU (and GPU)

Measurements and the reasoning behind the port are in [artifacts/report.md](artifacts/report.md).

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   (only when images are used)
```
That is all that is needed to run. Downloading the model also works with the standard library only.<br>
Environments are set up with [uv](https://docs.astral.sh/uv/).<br>
Optional dependency groups:
- `verify`: `tokenizers`, for the tokenizer cross-check (`check_tokenizer_fuzz`)
- `webcam`: OpenCV, for `demo_inference_webcam.py`
- `export`: torch, transformers and so on, to export the ONNX yourself (never used at run time)

# Installation

```bash
git clone https://github.com/Kazuhito00/Intern-Decision-0.8B-ONNX-CPU
cd Intern-Decision-0.8B-ONNX-CPU

# Run-time dependencies only (onnxruntime, numpy, pillow)
uv sync
```

# Download Model
Intern-Decision has no published ONNX. The exported ONNX files are on the [Releases](https://github.com/Kazuhito00/Intern-Decision-0.8B-ONNX-CPU/releases) page (they are not in the repository).<br>
The script below fetches them into `models/`. It uses plain HTTPS from the standard library: no git, no git-lfs.
```bash
uv run download_model.py
```
- **With no arguments it fetches every file of the release (all four graphs L=384 / 512 / 1024 / 1280, the vision encoder and the tokenizer: 17 files, about 4.1 GB)**
- It fetches from the latest release of `Kazuhito00/Intern-Decision-0.8B-ONNX-CPU` by default (use `--repo OWNER/NAME` or the environment variable `INTERN_ONNX_REPO` for another repository)
- Add `--seq-len 384` when only the runtime's default sequence length L=384 is needed (14 files, about 3.4 GB; see below)
- Every file is checked against the SHA-256 values listed in the release's `manifest.json` (a mismatch deletes the file and stops)
- Files that were split to stay under GitHub's limit (below 2 GiB per file) are joined after the download and checked as a whole
- Nothing is fetched again when the files are already there; an interrupted download continues where it stopped

```bash
uv run download_model.py --list                                          # list the files
uv run download_model.py --exclude "vision_encoder*"                  # text only (skip the vision encoder, 0.4 GB less)
uv run download_model.py --tag v0.0.0                                    # pick a tag (default: latest)
```

The release holds four graphs: L=384 (default), 512, 1024 and 1280 (about 4.3 GB in all). **They share one weights file (`weights.onnx.data*`, about 3.0 GB), so each extra bucket adds only its graph (about 0.25 GB).** To fetch just the default L=384, use either form (both give the same 14 files, about 3.4 GB):
```bash
uv run download_model.py --seq-len 384
uv run download_model.py --exclude "model_L512*" --exclude "model_L1*"
# spelled with --include, the shared files must be listed too
uv run download_model.py --include "*L384*" --include "weights*" --include "vision*" --include "*.json" --include "hf/*"
```
To add L=512 later, run it with `--seq-len 512` (weights already on disk are not fetched again).

To export it yourself (fetches the original checkpoint and writes the ONNX; needs torch):
```bash
uv sync --group export
uv run export/prepare_model.py                  # default: L=384
uv run export/prepare_model.py --seq-len 512    # a further bucket (the weights are shared; only the graph is new)
uv run export/prepare_model.py --seq-len 1024   # for more questions per request
```

# Usage

### Example (Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
The image demo uses `artifacts/images/sample_chart.png` by default (a bar chart; it asks "which bar is the tallest?" and "is this a bar chart?"). The sample is one picture of the synthetic data of the Wikipedia image benchmark (drawn by a script, `syn-0001`): not a photograph and nothing from Wikipedia. Use `--image` for your own pictures.
```bash
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```
**Every image is scaled down to 224 px on its long edge before it is given to the model** (`--size`, default 224; `0` = no resize). The graph is fixed at L tokens (default L=384): a 448 px picture needs about 570-600 tokens and does not fit L=384 (the engine switches to the L=1024 graph if it exists), a 224 px one needs about 64.

```python
from intern_decide import DecisionEngine

engine = DecisionEngine("models/onnx", tokenizer_json="models/hf/tokenizer.json")   # default L=384 (pass seq_len=512 etc. to change)

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

**All fields are decided in one forward pass.** The assistant turn is a JSON skeleton whose values are `<decision>` markers, and the logits **one position before** each marker decide that field. Nothing is generated.

Images go into the same request (up to 8):
```python
engine.predict({"state": "Look at the screenshot.", "images": ["shot.png"],
                "questions": {...}})
```
`images` accepts a file path, a `data:image/...;base64,` URL, bytes, or a Pillow image.

### Is there a person in the camera image?
Grabs frames with OpenCV, decides whether a person is in the newest frame and draws the answer on the live image.
```bash
uv run --group webcam demo_inference_webcam.py                        # camera 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/sample_chart.png   # an image or a video file instead of a camera (no person in it: NO)
```
- One decision takes a few seconds on the CPU, so it runs in its own thread and the live image never stops. Each time a decision arrives, the banner at the top (green = yes, red = no) and the probability of "yes" are updated
- Change the question with `--question` (a yes/no question). The picture given to the model is scaled to 224 px on its long edge (`--size`)
- Press `q` or `Esc` to quit. OpenCV is a dependency of this demo only

### Question types
| `type` | Answer |
|---|---|
| `choice` | one of the options (`choice`, `confidence`, `probabilities`) |
| `score` | the expected value on an ordinal scale (`score`, `legend`); the first level is the lowest |
| `noul` | whether a statement holds (`noul` = probability of yes); the options are fixed to `yes` / `no` |

### Running on a GPU (optional)
`--gpu` runs on CUDA (or DirectML). If this onnxruntime build has no GPU provider, or the model does not fit on the GPU, a warning is printed and the CPU is used.
```bash
uv run --no-sync demo_inference_text.py --gpu
uv run --no-sync verify/bench.py --gpu
```
```python
from intern_decide import DecisionEngine, gpu_providers

engine = DecisionEngine("models/onnx", tokenizer_json="models/hf/tokenizer.json", providers=gpu_providers())
print(engine.provider)   # the provider that is actually used
```

Notes:
- Install `onnxruntime-gpu` (CUDA; CUDA 13 and cuDNN 9 are needed) or `onnxruntime-directml` (Windows; no CUDA install needed) instead of `onnxruntime`. They write into the same `onnxruntime/` directory, so do not keep two of them. After switching, reinstall with `uv pip install --force-reinstall --no-deps onnxruntime-directml`
- **For CUDA, put the CUDA 13 and cuDNN 9 `bin` folders on `PATH`.** `onnxruntime-gpu` does not bundle the CUDA DLLs; it looks for them on `PATH`, and installing them is not always enough (`CUDA_PATH` may still point to an older CUDA). When they are not found, an error says `cublasLt64_13.dll` is missing and the CPU is used. A Windows example (adjust the versions):
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
  Open a new terminal after changing `PATH`
- `uv run` reinstalls the CPU `onnxruntime` to match `pyproject.toml`, so use `uv run --no-sync` (or `UV_NO_SYNC=1`)
- The provider that was actually used is shown on the last line of the demo and in the output of `verify/bench.py` (`provider`)
- The fp32 graph needs about 3.3 GB for its weights alone. On a GPU with little VRAM it does not fit and the CPU is used instead. See the performance section
- Probabilities on the GPU do not match the CPU exactly (a different order of operations). The comparison with the reference implementation is done on the CPU only

# The sequence length is fixed (important)
The model's linear attention walks 64-token chunks in a Python loop (`for i in range(num_chunks)`), which is **unrolled into a fixed number of steps at export time**. A graph with a dynamic sequence length cannot be built.

So the graph is exported at a fixed length L and every request is **right-padded** to L. All read positions come before the padding and the model is causal, so the result does not change (measured 1.9e-05). **The cost follows L, not the number of real tokens.**

Four buckets are exported: L=384, 512, 1024 and 1280, and **the default is L=384** (`DecisionEngine(..., seq_len=384)`; the demos and `verify/bench.py` take `--seq-len`). The prompt for one picture at 224 px plus one question is 260-375 tokens, so L=384 fits 100 % of them (L=256 fits under 20 %, L=320 94-99 %). The 2x1875 eval prompts (224 px) also fit L=384 in 100 % of cases.

- `seq_len` loads `model_L{seq_len}.onnx`; if it does not exist, the next larger bucket that does is used
- If a request does not fit, the engine switches to **the smallest bucket that exists and fits** (the old graph is released first, so two big graphs are never in memory; the vision encoder stays loaded). It then stays on that bucket
- Only when no existing bucket fits is a `ValueError` raised, naming the L that is needed
- An explicit `graph="model_L512.onnx"` wins and is never swapped (`graph_meta.json` keeps working as before)
- The four graphs **share one weights file** (`models/onnx/onnx/weights.onnx.data*`; tensors with identical bytes are read from the same place, so no number changes)

| L | questions that fit | one request (2 text questions, warm, median of 5) | session load | peak RAM |
|---|---|---:|---:|---:|
| **384 (default)** | up to 4 (one 224 px picture + 1 question also fits) | **2.23 s** | 32 s | 5.4 GB |
| 512 | up to 6 | 3.07 s | 32 s | 6.1 GB |
| 1024 | up to 13 | 6.13 s | 38 s | 10.6 GB |
| 1280 | 16 (the protocol limit) | 7.66 s | 36 s | 13.8 GB |

Measured on a Core i7-12800H, CPU, 6 threads, fp32, with `uv run --no-sync verify/bench.py --seq-len N --questions 2 --runs 5` (253-256 tokens). Heavy jobs were serialized with a lock (`heavy.lock`), but light jobs may have been running on the machine. The question counts are a guide for the sample state text (about 110 tokens) plus choice questions.

# Verification
```bash
uv run verify/run_all.py                                 # the gates that need the run-time dependencies only
uv run --group verify verify/run_all.py --full-fuzz      # run the fuzz in earnest when tokenizers is available
```

| Gate | What it checks | Needs |
|---|---|---|
| check_prompt | prompt string, `input_ids` and read positions match exactly | run-time dependencies |
| check_buckets | bucket choice, switching when a request does not fit, the `ValueError` when none fits, `graph=` winning | run-time dependencies |
| verify | decisions and probabilities agree (10/10); `--seq-len N` checks each bucket | run-time dependencies |
| check_image | the preprocessing matches the processor's `pixel_values` | + pillow |
| verify_images | image decisions agree (4/4) | + pillow |
| check_tokenizer_fuzz | cross-check against the Rust `tokenizers` | + tokenizers |
| check_padding | right padding does not move the read logits | + torch |
| probe_lengths | measures why the length has to be fixed | + torch |

`artifacts/golden.json` and `golden_images.json` are ground truth made by actually running the reference implementation (PyTorch fp32, CPU). Both run the same weights, so any difference is a bug in the port.

### Performance (Core i7-12800H, CPU only, fp32, L=512; per-bucket numbers are in the table of "The sequence length is fixed")
| Item | Value |
|---|---:|
| 2 text questions | 3051 ms |
| 1 image + 2 questions | 3296 ms |
| Session load | 32 s |
| Peak RAM | 5.8 GB |

8 threads is the best setting.

### GPU measurements (NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM, `verify/bench.py`, 2 text questions, 253 tokens)
| Run | Median | Session load | Notes |
|---|---:|---:|---|
| CPU (6 threads) | 2990 ms | 37 s | peak RAM 6.1 GB |
| GPU (CUDA, `onnxruntime-gpu` 1.31, CUDA 13.4, cuDNN 9.20) | 3619 ms | 48 s | 3867 / 4096 MiB of VRAM used, GPU utilization 97-98 % |
| GPU (DirectML, `onnxruntime-directml` 1.24) | - | - | initialization failed; a warning was printed and the CPU was used |

- **On this GPU it is not faster than the CPU.** The fp32 graph needs about 3.3 GB for its weights alone, which nearly fills the 4 GB of VRAM. It is not a warm-up effect: 16 runs of the same input take a steady 3.1-3.25 s from the second run on. One forward pass runs about 22,600 tiny operations (the unrolled linear attention); their GPU time adds up to about 0.57 s while the wall time is about 3.1 s, so launching the operations one by one dominates and the GPU's compute is not used (about 1 % of the nodes ran on the CPU). A GPU with more VRAM was not measured
- The first measurement was 8427 ms, but other heavy jobs were running at the same time. The 3619 ms above is the repeat
- DirectML failed to initialize with a `'utf-8' codec can't decode` error (the cause was not investigated; the message is a character-encoding error). The code prints a warning and switches to the CPU

# Limitations
- The sequence length is fixed. The default is L=384 and a longer request switches to a larger bucket; beyond the largest one it raises an exception (it names the `--seq-len` that is needed)
- fp32 only. Video is not supported
- `tokenizer.py` / `chat.py` are specific to this checkpoint and reject unexpected input with an exception
- Do not export with `optimize=True` (it does not finish in over 70 minutes for 24 layers)
- On a GPU the probabilities do not match the CPU exactly, and with little VRAM the model does not fit and the CPU is used

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (run time: onnxruntime, numpy, pillow; groups: verify / webcam / export)
download_model.py        # fetch the model (GitHub Releases, SHA-256 check, joins split files)
demo_inference_text.py   # text inference demo
demo_inference_image.py  # image inference demo
demo_inference_webcam.py # "is there a person?" on a camera image (OpenCV)
intern_decide/           # the inference implementation (light dependencies only; never imports export/)
  runtime.py             #   ORT session + right padding + read-out + GPU providers
  tokenizer.py           #   pure-Python Qwen byte-level BPE
  protocol.py            #   inference.py's schema, temperature scaling and argmax, ported
  chat.py                #   chat template (no jinja2) and 3-D M-RoPE
  image.py               #   Pillow decode + smart_resize + patching
  _uniprops.py           #   generated: Unicode range tables
export/                  # heavy dependencies (torch / transformers / onnx)
  prepare_model.py       #   fetch the checkpoint + write the ONNX
  export_onnx.py         #   single graph (read-out gather built in) + vision encoder
  share_external_data.py #   buckets share one weights file (identical tensors only)
  make_golden.py         #   makes artifacts/golden.json
  vision_export_patch.py #   swaps in the single-image version
release/
  make_release.py        # builds the release files from models/ (split, SHA-256, manifest.json)
verify/                  # verification gates (run them all with run_all.py)
artifacts/               # golden.json (ground truth for the checks), report.md
models/                  # ONNX, tokenizer and config (gitignored)
```

# Reference
- [internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) (the original model; Shanghai AI Laboratory)
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (the base of the original model; Alibaba Cloud)

# License
Intern-Decision-0.8B-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
The model weights follow the license of the original model ([internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B), Apache-2.0; the Qwen3.5 base is Apache-2.0 too). Both license texts (`LICENSE`, `LICENSE-QWEN`) are included in the release, unchanged.

## About the converted files
The ONNX files in the release are a conversion of the checkpoint of [internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) (Shanghai AI Laboratory; Apache-2.0), for which no ONNX is published, made with this repository's `export/export_onnx.py` (the graph is exported, and the weights are shared between the graphs of different sequence lengths). The weights are exported in fp32, and the decisions were checked against the original implementation (PyTorch fp32): text 10/10, images 4/4, largest probability difference 1.7e-6. The base model is [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Alibaba Cloud; Apache-2.0). The providers of the original model do not endorse this conversion or warrant how it behaves.
