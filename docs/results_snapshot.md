# PTQ Results Snapshot

This file captures the current cleaned-core benchmark results for the rotor study.

Source protocol:
- [study_protocol.md](study_protocol.md)
- [results_template.md](results_template.md)

## Run Metadata

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Environment | `conda run -n scabi` |
| Device | CUDA |
| Models | `gpt2`, `gpt2-medium` |
| Method | Learned block-diagonal rotor |
| Core settings | `--n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64` |

## Core Metrics

| Model | FP16 PPL | RTN PPL | Quantized PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 87.56 | 4.6% | 46/48 | 123.1s |
| `gpt2-medium` | 44.17 | 55.19 | 55.16 | 4.4% | 89/96 | 408.6s |

## Grade Allocation

Current live slice:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex 'mlp\.c_proj$' --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex 'mlp\.c_proj$' --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

| Model | FP16 PPL | RTN PPL | Grade Allocation PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 88.07 | 59.4% | 12/12 | 0.8s |
| `gpt2-medium` | 44.17 | 55.19 | 53.21 | 58.7% | 24/24 | 1.7s |

Broad slice used as a comparison point:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex '(?:attn|mlp)\.c_proj$' --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex '(?:attn|mlp)\.c_proj$' --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

| Model | FP16 PPL | RTN PPL | Grade Allocation PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 95.49 | 35.0% | 22/24 | 1.1s |
| `gpt2-medium` | 44.17 | 55.19 | 53.60 | 27.5% | 38/48 | 2.2s |

Attention-only slice used for rejection:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex 'attn\.c_proj$' --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex 'attn\.c_proj$' --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

| Model | FP16 PPL | RTN PPL | Grade Allocation PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 102.18 | 8.7% | 10/12 | 0.8s |
| `gpt2-medium` | 44.17 | 55.19 | 55.36 | -5.2% | 14/24 | 1.4s |

## Grade Allocation Interpretation

| Question | Notes |
|---|---|
| Did the `mlp.c_proj` slice improve perplexity? | Yes on both models, though the `gpt2` gain is modest. |
| Did the broad slice improve perplexity? | Yes on `gpt2-medium`, no on `gpt2`. |
| Did the attention-only slice improve perplexity? | No. It regressed on both models. |
| Did NMSE improvement predict perplexity? | Only partially. Local gains were real, but they did not transfer perfectly. |
| Was the transform cheap enough? | Yes. It was much cheaper than the rotor-scale or projection branches. |
| Does this justify more work? | Yes, on the `mlp.c_proj` slice. The attention-only slice is a rejection, and the broad slice is now just a comparison point. |

## Diagonal Scaling Baseline

Stable settings used:

```bash
python -m gaptq.quantize_model --experimental --diag-scale --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --diag-scale --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

| Model | FP16 PPL | RTN PPL | Diagonal Scaling PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 91.69 | 6.4% | 8/12 | 0.3s |
| `gpt2-medium` | 44.17 | 55.19 | 55.94 | -3.3% | 13/24 | 0.6s |

## Diagonal Scaling Interpretation

| Question | Notes |
|---|---|
| Did diagonal scaling beat RTN? | Yes on `gpt2`, no on `gpt2-medium`. |
| Was it cheaper than the geometric transforms? | Yes. It was very cheap. |
| Did it give a cleaner answer than grade allocation? | No. It looks like a useful comparator, not the main method. |
| What does it tell us? | Non-GA diagonal preconditioning is a real baseline, but it is not enough by itself to replace the current grade-allocation result. |

## Reflection Baseline

Stable settings used:

```bash
python -m gaptq.quantize_model --experimental --reflection --model gpt2 --n-bits 4 --n-steps 1 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64 --skip-rtn
python -m gaptq.quantize_model --experimental --reflection --model gpt2-medium --n-bits 4 --n-steps 1 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64 --skip-rtn
```

| Model | FP16 PPL | RTN PPL | Reflection PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 92.26 | 0.0% | 30/48 | 0.6s |
| `gpt2-medium` | 44.17 | 55.19 | 57.05 | 0.0% | 65/96 | 1.8s |

## Projection + Residual Model

Stable settings used:

```bash
python -m gaptq.quantize_model --experimental --projection-residual-model --model gpt2 --n-bits 4 --n-steps 1 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --projection-residual-model --model gpt2-medium --n-bits 4 --n-steps 1 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

| Model | FP16 PPL | RTN PPL | Projection + Residual Model PPL | Mean NMSE gain vs RTN | Layers improved | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 2106.30 | -682.7% | 11/24 | 238.3s |
| `gpt2-medium` | 44.17 | 55.19 | 2573.19 | -1240.7% | 0/48 | 974.9s |

## Layer Notes

| Layer family | Observation |
|---|---|
| Attention | Often modest NMSE gains, but some projections remain sensitive. |
| MLP | Largest and most consistent NMSE gains, especially in expansion projections. |
| Early layers | Generally improve, though some projections can regress slightly. |
| Mid layers | Stable improvements in most runs. |
| Late layers | More mixed; some negative NMSE deltas appear near the top of the stack. |

## Interpretation

| Question | Notes |
|---|---|
| Did the method improve perplexity? | Yes on `gpt2`, essentially tied on `gpt2-medium`. |
| Did NMSE improvement predict perplexity? | Only partially for the rotor path. Reflection did not translate into better perplexity. |
| Was the transform cheap enough? | Reflection was cheap, but not useful end to end. |
| Would this justify scaling / clipping / more grouping? | Yes. The rotor+scaling path is the more credible hybrid direction. |

| Question | Notes |
|---|---|
| Did the method improve perplexity? | No. It catastrophically worsened perplexity on both models. |
| Did NMSE improvement predict perplexity? | No. The low-rank residual correction overfit the calibration objective. |
| Was the transform cheap enough? | No. It was much slower than the rotor path and still failed. |
| Would this justify more work on the same branch? | No. Keep it archived unless the residual objective changes materially. |

## Local Notes

- Keep machine-specific `scabi` commands in [local_scabi.md](local_scabi.md).
- Do not treat this file as the final result section; it is a living snapshot.
