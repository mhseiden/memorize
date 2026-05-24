# memorize-embed

Text-embedding wrapper for memorize. Wraps `fastembed` (which wraps `ort` 2.0)
and adds a token-cap re-splitter so callers can stay under the model's
positional-embedding window.

Default model: `sentence-transformers/all-MiniLM-L6-v2` (fp32, 384-d, 512 max
tokens). Override with `MEMORIZE_EMBED_MODEL=minilm|minilm-q|gte-large|jina-code`.

## Why this backend (and not something else)

Each row below was measured against the synthetic batch sweep in
`memorize bench embed` on an M4 Pro / macOS 15. Numbers are peak chunks/s
with 800-char synthetic inputs (~250 tokens each).

| backend                        | peak chunks/s | per-chunk | notes |
|--------------------------------|---------------|-----------|-------|
| **fastembed/ORT, fp32 MiniLM** | **166** (batch=4) | ~6 ms | current default |
| fastembed/ORT, INT8 MiniLM-Q   | 143 (batch=16)    | ~7 ms | slower on ARM — see below |
| Candle 0.10 + Metal            | 94 → 54 (decays) | 10–18 ms | broken batch dim — see below |
| Candle 0.10 + Accelerate (BLAS)| 25 (flat)        | ~42 ms | unexpectedly slow forward pass |
| Candle 0.10 + plain CPU        | ~12              | ~80 ms | reference for the above |
| ORT + CoreML EP                | ~10× slower than CPU EP | — | graph fragmentation, see below |

### INT8 quantization (MiniLM-Q)

Public guidance for INT8 dynamic quantization advertises 2–4× CPU speedup
vs fp32. That's **x86 + VNNI only**. On Apple Silicon, ORT's ARM INT8 path
falls back to emulated kernels and runs ~15% **slower** than fp32. The Q
variant is kept available as a non-default option in case anyone runs this
on an x86 box.

### Candle + Metal

Per-call latency from a traced run, BERT MiniLM forward pass:

```
batch=1   seq~250   fwd~10 ms   (10 ms / chunk)
batch=8   seq=276   fwd~104 ms  (13 ms / chunk)
batch=32  seq=276   fwd~485 ms  (15 ms / chunk)
batch=128 seq=277   fwd~2256 ms (17.6 ms / chunk)
```

Forward time scales **linearly with batch size**. Candle 0.10's Metal
backend isn't running a real batched matmul — it iterates per item
internally — so GPU parallelism is wasted and we eat extra padding
overhead from `BatchLongest` on top. Tracked in upstream Candle issues
#1062 (BERT slower than HF on M1) and #2659 (Metal perf regression).

If a future Candle release lands a properly batched Metal matmul for
BERT, this is worth revisiting — the kernel-launch cost (~50 µs / op,
~120 ops per BERT forward = ~6 ms baseline) is the only structural
floor.

### Candle + Accelerate

Surprising loss: ~6× slower than ORT despite Accelerate giving us
Apple's optimized BLAS. Suggests Candle's BERT graph has overhead beyond
the matmul itself (many small per-op kernel calls). Did not investigate
further — the result was decisive.

### CoreML EP

Tested via `MEMORIZE_EMBED_COREML=1` (still wired into this crate, see
the module docs for env flags). On MiniLM specifically:

- Compiles cleanly under MLProgram (NeuralNetwork backend crashes on
  macOS 14+).
- Inference is ~10× **slower** than CPU EP.
- Root cause is well-documented: ORT's CoreML EP fragments transformer
  graphs into many CPU↔ANE partitions; per-batch marshaling + dynamic-
  shape reshape dominate compute at our batch sizes. ORT issue #28022
  covers the partition-round-trip problem; the diagnostic literally
  prints `CoreML is not recommended with this model` for transformers.

Kept opt-in only.

## What we'd reach for if we needed more throughput

These are not implemented; they're recorded so the next investigator
doesn't redo the search.

1. **llama.cpp / ggml BERT encoder.** Has proper Metal kernels for
   transformer inference, proven fast on Apple Silicon. Integration is
   C-FFI with build-system complexity; estimated 3–5 days.
2. **x86 + AVX-512 / VNNI hardware.** INT8 quantization would actually
   pay off, and ORT's optimized AVX paths apply. Trivial: just run on
   a different box; no code change.
3. **Wait for Candle.** When Candle ships a batched Metal matmul for
   BERT-family models, the prototype we cut would beat ORT.

For now, the practical move is to live with the ORT ceiling. Cold-scan
is a one-time cost; the file watcher does incremental updates from
there, so absolute throughput on big repos matters less than it looks.
