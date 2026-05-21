# External PTQ Comparison

This note compares the current repo direction against the strongest external PTQ
families that matter for the same problem.

The short version:

- **GPTQ** is the strongest weight-only baseline to compare against when you care about accuracy-sensitive 3-4 bit PTQ.
- **AWQ** is the closest external analogue to the repo’s current `mlp.c_proj` grade-allocation work, because it is activation-aware and still weight-only.
- **SmoothQuant** is the closest analogue to the diagonal-scaling comparator, because it explicitly migrates activation difficulty into the weights with an equivalent transform.
- **QuaRot** is the closest external analogue to the earlier rotor work, but it is broader: rotation is only one part of an end-to-end 4-bit inference pipeline.

The repo now contains an AWQ-style proxy baseline on `mlp.c_proj` that behaves like a
saliency-weighted mixed-precision comparator. It is not a full AWQ implementation,
but it is the right external-style baseline to compare against the current live
grade-allocation branch.

## 1. Method Summary

| Method | Core idea | What it is good at | What it suggests for this repo |
|---|---|---|---|
| GPTQ | Approximate second-order, one-shot weight quantization | Strong weight-only accuracy at 3-4 bits | Uniform RTN is too weak a baseline; sensitivity-aware weight PTQ should be compared next |
| AWQ | Activation-aware weight quantization | Low-bit weight-only quantization with activation/outlier awareness | The repo’s grade allocation is conceptually closest to AWQ-style bit budgeting; the new AWQ proxy now serves as the strongest in-repo non-GA comparator |
| SmoothQuant | Smooth activation outliers by equivalent scaling | W8A8 PTQ and hardware-efficient inference | The repo’s diagonal-scaling baseline is the nearest in spirit, though not the same method |
| QuaRot | Rotation-based end-to-end 4-bit inference | Rotation plus activation/KV quantization | Rotor-only is not enough; the transform must fit a larger quantization pipeline |

## 2. What each method says about the current repo

### GPTQ

GPTQ uses approximate second-order information to decide how to quantize weights. The main lesson is that local reconstruction error matters, but not all weights matter equally. This is stronger than RTN because it accounts for sensitivity.

Implication for GAP-TQ:
- If grade allocation only beats RTN, that is not enough.
- The next fair comparison is GPTQ or a close approximation to sensitivity-aware weight PTQ.
- The repo should not read RTN as the real bar for success.

Sources:
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [Official GPTQ repository](https://github.com/IST-DASLab/gptq)

### AWQ

AWQ is activation-aware weight quantization. It searches for weight scaling that protects salient channels while keeping the method hardware-friendly for 3-4 bit weight-only inference.

Implication for GAP-TQ:
- AWQ is the closest conceptual comparator to the current `mlp.c_proj` grade-allocation branch.
- Both approaches are trying to spend precision unevenly on the weights.
- If AWQ beats grade allocation, then the repo’s current gain is mostly a heuristic improvement over RTN, not a new structural result.
- The in-repo AWQ proxy currently beats RTN on both `gpt2` and `gpt2-medium`, but grade allocation still wins on both models.

Sources:
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [Official AWQ repository](https://github.com/mit-han-lab/llm-awq)

### SmoothQuant

SmoothQuant is a mathematically equivalent scaling transform that reduces activation outliers by shifting difficulty from activations to weights. It is mainly a W8A8 method, not a weight-only 4-bit method.

Implication for GAP-TQ:
- The repo’s diagonal-scaling baseline is the closest current comparator.
- The lesson is that scaling can matter more than transform geometry by itself.
- If scaling helps but grade allocation does not generalize, the current gain may be mostly due to redistribution of ranges rather than GA structure.

Sources:
- [SmoothQuant paper](https://arxiv.org/abs/2211.10438)
- [Official SmoothQuant repository](https://github.com/mit-han-lab/smoothquant)

### QuaRot

QuaRot uses rotations to remove outliers and quantize the whole model end-to-end, including weights, activations, and KV cache.

Implication for GAP-TQ:
- The repo’s earlier rotor experiments were a partial version of this idea.
- QuaRot shows that rotation is more useful when it is part of a pipeline that also addresses activations and inference-time constraints.
- Rotor-only local gains are not enough.

Sources:
- [QuaRot paper](https://arxiv.org/abs/2404.00456)
- [Official QuaRot repository](https://github.com/z-lab/paroquant)

## 3. Practical conclusion for this repo

The external PTQ literature suggests the following order of importance:

1. Sensitivity-aware weighting or allocation
2. Activation outlier handling
3. Hardware-friendly scaling / reparameterization
4. Rotation as a helper, not the entire method

That lines up with the repo’s empirical path:

- rotor-only helped local error but was not consistently enough for perplexity
- diagonal scaling helped `gpt2` but not `gpt2-medium`
- the AWQ proxy is the strongest in-repo external-style comparator and beats RTN on both models
- grade allocation on `mlp.c_proj` is still the best live result

So the next benchmark question is no longer whether a stronger external-style comparator exists; it does, and the repo now has one. The open question is whether grade allocation is still genuinely better than the AWQ-style baseline, or whether it is just a heuristic improvement over RTN.
