# memorize-embed

Text-embedding wrapper for memorize. Wraps `llama-cpp-2` (which links
llama.cpp) and exposes a tokenizer-driven splitter so callers can stay
under the model's positional-embedding window.

Default model: `sentence-transformers/all-MiniLM-L6-v2`, F32 GGUF from
`leliuga/all-MiniLM-L6-v2-GGUF`, 384-d, 512 max tokens. Override:

```text
MEMORIZE_EMBED_GGUF_REPO  default: leliuga/all-MiniLM-L6-v2-GGUF
MEMORIZE_EMBED_GGUF_FILE  default: all-MiniLM-L6-v2.F32.gguf
```

## Why this backend (and not the other things we tried)

All numbers below are from `memorize bench embed` on an M4 Pro / macOS 15,
synthetic 800-char inputs (~250 tokens), peak chunks/s across the
batch-size sweep. Reports under `bench/`.

| runtime + model                  | peak chunks/s | vs ORT-CPU baseline |
|----------------------------------|---------------|---------------------|
| **llama.cpp Metal + MiniLM F32** | **442** (batch=4) | **2.66×** ← chosen |
| llama.cpp Metal + MiniLM Q8      | 441 (batch=4)     | 2.66× (tied) |
| llama.cpp Metal + BGE-small Q8   | 233 (batch=4)    | 1.40× |
| llama.cpp Metal + BGE-M3 FP16    | 32 (batch=4)     | 0.19× |
| llama.cpp Metal + Jina-code-0.5B Q8 | 21 (batch=2)  | 0.13× |
| llama.cpp CPU + MiniLM F32       | 127               | 0.77× |
| fastembed/ORT-CPU + MiniLM F32   | 166 (batch=4)    | 1.0× (baseline) |
| fastembed/ORT-CPU + MiniLM-Q INT8 | 143 (batch=16)  | 0.86× |
| MLX (Python) + MiniLM            | 267 (batch=32)   | 1.61× |
| Candle Metal + MiniLM            | 94 → 54 (decays)  | 0.57× |
| Candle Accelerate + MiniLM       | 25 (flat)         | 0.15× |
| ORT + CoreML EP                  | ~10× slower than CPU EP | — |

### llama.cpp + Metal — what we ship

Peak 442 chunks/s at batch=4. The throughput curve shape is:

```
batch=1  →  363   (no batch amortization)
batch=2  →  429
batch=4  →  442   ← peak
batch=8  →  393
batch=16 →  333
batch=32 →  249
batch=64 →  154
batch=128 →  78
batch=256 →  30
```

Sweet spot is batch 2–8 — fortunately the median file produces ~6 chunks,
so the production indexer naturally lives at the peak. **Cross-file
batching is a trap on this hardware**: padding waste and per-batch GPU
dispatch cost overwhelm the marginal kernel-launch savings past batch=16.

Slate ~113k chunks at typical batch=5–8 → ~4.7 min cold-scan (vs ~11 min
on ORT CPU).

Known wart: the process exits with SIGABRT after the report is fully
written, from a llama.cpp ggml-metal teardown assert
(`GGML_ASSERT([rsets->data count] == 0)`). Tracked upstream as PR #17869.
Cosmetic — doesn't affect output. Live with it until upstream lands a fix.

### What we rejected and why

**fastembed/ORT-CPU (previous default).** Mature, fast on x86 CPU, easy
Rust integration. On Apple Silicon: 166 chunks/s ceiling. ORT's CPU EP
has no Metal equivalent.

**ORT INT8 quantization (MiniLM-Q).** Public guidance cites 2–4× CPU
speedup over fp32. That's **x86 + VNNI only**. On Apple Silicon, ORT's
ARM INT8 path falls back to emulated kernels and runs ~15% *slower* than
fp32. Same finding repeated for llama.cpp Q8 vs F32 (tied, no speedup;
the win is 3.5× smaller GGUF file, not throughput).

**ORT CoreML EP.** Compiles cleanly under MLProgram, but transformer
graphs fragment into 14+ CPU↔ANE partitions; per-batch marshaling
dominates. Result: ~10× *slower* than CPU EP. ORT's own diagnostic
prints `CoreML is not recommended with this model` for transformers
(issue #28022).

**Candle 0.10 + Metal.** Forward time scales **linearly with batch
size** — Candle's Metal backend doesn't run a real batched matmul, it
loops per item. Per-call trace:

```
batch=1   seq~250   fwd~10 ms   (10 ms / chunk)
batch=8   seq=276   fwd~104 ms  (13 ms / chunk)
batch=32  seq=276   fwd~485 ms  (15 ms / chunk)
batch=128 seq=277   fwd~2256 ms (17.6 ms / chunk)
```

Tracked in Candle issues #1062 (BERT slower than HF on M1) and #2659
(Metal perf regression). Revisit if Candle ships a properly batched
Metal matmul.

**Candle + Accelerate (BLAS).** ~6× slower than ORT despite Apple's
optimized BLAS underneath. Candle's BERT graph has overhead beyond the
matmul itself (many small per-op kernel calls). Did not investigate
further — decisive loss.

**MLX (Python via mlx-embeddings).** 267 chunks/s peak, but in Python
with `mx.eval()` synchronization and a curious cliff at batch=128. No
mature first-class Rust binding — would have to write C++/FFI ourselves
or shell out to Python. 4–5 days of integration work for a result that's
already 25% slower than llama.cpp.

**llama.cpp + CPU only** (verifying the Metal speedup is real). 127
chunks/s — slightly *slower* than ORT CPU. So llama.cpp's CPU kernels
aren't competitive with ORT on x86-style workloads; the win is entirely
Metal.

### What we'd reach for if we needed more throughput

These are not implemented; they're recorded so the next investigator
doesn't redo the search.

1. **x86 + AVX-512 / VNNI hardware.** INT8 quantization would actually
   pay off, and ORT's optimized AVX paths apply. Trivial: run on a
   different box; no code change.
2. **Wait for Candle.** When Candle ships a batched Metal matmul for
   BERT-family models, the prototype we cut (since removed) would beat
   llama.cpp on simpler Rust integration.
3. **A code-specialized small embedder.** Jina-code-0.5B looked
   promising on the spec sheet but is 20× slower than MiniLM-L6 on this
   hardware. Smaller code-specialized embedders may emerge — worth
   re-benching every 6 months.

For now, MiniLM-L6 + llama.cpp Metal is the ceiling on Apple Silicon
for our workload. Cold-scan is a one-time cost; the file watcher does
incremental updates from there, so absolute throughput on big repos
matters less than it looks.
