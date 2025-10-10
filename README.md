# <p align="center">roketmodel</p>

This page focuses **only on the model** and provides detailed technical documentation. For training data, CLI, and RAG-lite details, see the main README.

> **Privacy Note:** This project contains **NO** organization names or confidential information. The entire dataset was produced from scratch by the developer (term–definition lists and attribute files). **NO** corporate or proprietary data was used.

---

## Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Sizing & Complexity](#sizing--complexity)
* [I/O Contracts](#io-contracts)
* [Tokenizer](#tokenizer)
* [Training Settings](#training-settings)
* [Inference (Runtime)](#inference-runtime)
* [Accelerations & Optimizations](#accelerations--optimizations)
* [Evaluation Metrics](#evaluation-metrics)
* [Checkpoint Format](#checkpoint-format)
* [Python API Examples](#python-api-examples)
* [Configuration Reference](#configuration-reference)
* [Limitations & Known Issues](#limitations--known-issues)
* [Versioning & Compatibility](#versioning--compatibility)
* [FAQ](#faq)

---

## Overview

* **Model type:** Character-level **decoder-only Transformer** (causal LM head).
* **Focus domain:** Defense terminology and concise technical definition generation.
* **Training strategy:** Two-stage — (1) Causal LM, (2) Instruction Fine-Tuning (Q&A alignment).
* **Augmentation:** Optional **RAG-lite** context injection to enrich prompts.

> Note: All sections here describe the model’s behavior and interface only.

---

## Architecture

The following outlines a lean decoder-only stack:

```bash
flowchart LR
  T[Tokenizer (char-level)] -->|ids| E[Token Embedding]
  PE[Positional Encoding] --> E
  E --> B1[Block × N]
  subgraph Decoder Block
    MHA[Masked Multi-Head Self-Attention]
    LN1[LayerNorm (pre)]
    FFN[Feed-Forward (GELU)]
    LN2[LayerNorm (pre)]
  end
  B1 -->|logits| LM[Causal LM Head]
````

**Decoder Block:**

* Pre-LayerNorm with residual connections.
* **MHA:** causal (upper-triangular masked) attention.
* **FFN:** two linear layers + **GELU** activation; optional dropout.

> Positioning: sinusoidal or learned positional representation (configurable).

---

## Sizing & Complexity

The values below reflect the default configuration; adjust via `configs/model.yaml`.

* **d\_model:** 512
* **# Layers (N):** 8
* **# Heads (n\_heads):** 8 (head dim = 64)
* **FFN width (d\_ff):** 2048
* **Dropout:** 0.1
* **Vocab:** Character-based (ASCII + Turkish extensions + special symbols)
* **Context window (max\_len):** 1024 (recommended)

**Rough parameter count (back-of-the-envelope):**

* Embedding: `vocab_size × d_model`
* MHA (per layer): `4 × d_model × d_model` (Q, K, V, O projections)
* FFN (per layer): `2 × d_model × d_ff + d_ff × d_model`
* LM head: `d_model × vocab_size`

Total ≈ Emb + N × (MHA + FFN) + Head (bias and LN terms omitted).

**Time complexity:** O(L²·d) (full attention). For long prompts, `block-sparse` / `sliding window` is experimental (config flag).

---

## I/O Contracts

* **Input:** Character id sequence (int64), shape `(B, T)`
* **Output (training):** Logits `(B, T, vocab_size)` with shifted labels for causal loss.
* **Output (inference):** Generated text and optional `scores` (log-probabilities) / `attentions` (optional).

**Masks:**

* Causal upper-triangular mask: future positions are hidden.
* Padding mask: for shorter sequences (optional; typically minimal at char level).

---

## Tokenizer

* **Type:** Character-level, fixed vocabulary.
* **Normalization:** NFC (default), Unicode-safe escaping.
* **Special tokens:** `<BOS>`, `<EOS>`, `<PAD>` (optional), `<SEP>`.
* **Error tolerance:** Corner-case fixes for Turkish diacritics; guard against double BOS/EOS injection.

---

## Training Settings

* **Loss:** Causal LM (Cross-Entropy).
* **Optimization:** AdamW (weight decay=0.1; no decay on LN/Embedding).
* **LR:** 3e-4 (cosine annealing + warmup=2000 steps).
* **Epochs:** 107 (example setup).
* **Batch Size:** 64 (effective BS can be increased via **gradient accumulation**).
* **Regularization:** Dropout 0.1, **clip\_grad\_norm=1.0**.
* **EMA:** Optional weight EMA (recommended).

**Instruction FT:**

* Style: Q\&A pairs, diversified with negative/edge cases.
* Goal: Produce short, precise, instruction-aligned technical definitions.

---

## Inference (Runtime)

* **Sampling:** `temperature`, `top_p`, `top_k`.
* **Length:** `max_new_tokens` (recommendation: 128–256)
* **Stopping criteria:** `<EOS>` or max length.
* **Streaming:** Token-by-token streaming mode supported in the CLI.

**Example (Python):**

```python
from roketgpt.model import load_model
m = load_model("checkpoints/roketgpt_001/best.pt", device="cpu")
text = m.generate(
    prompt="Define: short-range, land-based anti-tank system",
    max_new_tokens=128,
    temperature=0.8,
)
print(text)
```

---

## Accelerations & Optimizations

* **KV cache:** Reduces per-step recomputation.
* **`torch.compile`** (optional): Graph capture for speedups.
* **Heavyweight compression paths:** weight-only int8; activations fp16/bf16 (flag-gated).
* **Block-sparse attention (experimental):** Cuts memory/time on long prompts.
* **CPU path:** Allocation-reduced attention; measured ≈ *88 ms/token* (on example hardware).

> Note: Actual speed depends on hardware and the PyTorch build.

---

## Evaluation Metrics

* **Exact Match (EM):** 97.4%
* **Token Accuracy:** 82.1%
* **Validation Loss:** 0.0000
* **Latency:** ≈ 88.08 ms/token (CPU)

Metric computation is described in `src/roketgpt/eval.py`.

---

## Checkpoint Format

* **Files:** `best.pt` / `last.pt`
* **Contents:**

  * `model_state_dict`
  * `optimizer_state_dict` (for training)
  * `scaler` (if AMP is used)
  * `config` snapshot (seed, hyperparameters)
  * `git_sha` and timestamp (if available)

Load safety: use `strict=False` for missing/extra params.

---

## Python API Examples

**Load model:**

```python
from roketgpt.model import load_model
model = load_model(path="checkpoints/roketgpt_001/best.pt", device="cuda:0")
```

**Get logits:**

```python
logits = model.forward_ids(input_ids)  # (B, T, vocab)
```

**Text generation (advanced options):**

```python
out = model.generate(
    prompt, max_new_tokens=256, temperature=0.7, top_p=0.9,
    repetition_penalty=1.1, stop_tokens=["<EOS>"]
)
```

---

## Configuration Reference

```yaml
model:
  d_model: 512
  n_layers: 8
  n_heads: 8
  d_ff: 2048
  dropout: 0.1
  max_len: 1024
  pos_encoding: "learned"   # or "sinusoidal"
  layer_norm: "pre"
  init: "xavier_uniform"
  vocab: "char"
```

---

## Limitations & Known Issues

* Being character-level, **very long contexts** may degrade efficiency; RAG-lite is recommended.
* Long repetitive sequences (e.g., tables/numbers) can increase repetition risk; use `repetition_penalty`.
* For highly out-of-domain prompts, hallucination risk rises; prefer **source-grounded** responses.

---

## Versioning & Compatibility

* Module version: accessible via `roketgpt.__version__`.
* **SemVer**-compliant tags (e.g., `0.1.0`).
* Checkpoint backward compatibility is preserved across minor versions; a conversion script is provided for major bumps.

---

## FAQ

**Q: Can I run the model on CPU?**
A: Yes; if it’s slow, try `--compile false`, `--fp16 false`, and a smaller `max_new_tokens`.

**Q: Best settings for quality?**
A: `temperature∈[0.6,0.9]`, `top_p∈[0.8,0.95]`, `top_k=50`, paired with a good RAG-lite context.

**Q: How to handle long input?**
A: Chunk and summarize prompts, or enable the experimental block-sparse attention path.


