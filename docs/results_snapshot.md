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
| Did NMSE improvement predict perplexity? | Only partially. NMSE improved broadly, but perplexity gains were smaller. |
| Was the transform cheap enough? | It was feasible, but noticeably slower on `gpt2-medium`. |
| Would this justify scaling / clipping / more grouping? | Yes. The current result suggests a hybrid method is worth testing. |

## Local Notes

- Keep machine-specific `scabi` commands in [local_scabi.md](local_scabi.md).
- Do not treat this file as the final result section; it is a living snapshot.

