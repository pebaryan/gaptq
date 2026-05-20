# Early Literature Review: Post-Training Quantization for LLMs

This is an early review of the PTQ problem as it relates to this repo.

The goal is not to list every paper. The goal is to identify the main pressure points in PTQ and what each major family of methods is actually solving.

## 0. PTQ Methods At a Glance

| Method | Main lever | What it mainly fixes | Typical tradeoff | Relevance to rotors |
|---|---|---|---|---|
| RTN | Uniform rounding | Nothing structural; only baseline compression | Cheap but weak | Baseline only |
| GPTQ | Second-order sensitivity | Layer-wise weight sensitivity | Stronger calibration cost and complexity | Shows that local structure matters |
| SmoothQuant | Scaling / smoothing | Activation outliers | Moves difficulty into weights | Rotors may complement scaling |
| AWQ | Activation-aware weight selection | Important channels under low-bit weight-only PTQ | Needs activation statistics | Suggests layer/channel importance matters |
| QuaRot | Orthogonal rotation | Basis misalignment and hidden-state outliers | Extra transform step | Closest predecessor to rotor PTQ |
| ParoQuant | Pairwise rotation + scaling | Outliers and dynamic range | More moving parts, but still lightweight | Best modern reference point for rotors |

Recommended reading order:

1. GPTQ
2. SmoothQuant
3. AWQ
4. QuaRot
5. ParoQuant

That order follows the field’s progression from sensitivity-aware rounding, to scaling, to activation-aware weighting, to rotations, and then to rotation plus scaling.

## 1. What PTQ is optimizing

At a high level, PTQ tries to:

- preserve downstream task behavior
- reduce model size and memory traffic
- keep calibration cheap
- stay implementable on real hardware

The hard part is that these goals conflict. A method can be mathematically neat and still fail to preserve perplexity or runtime efficiency.

## 2. Main failure modes in low-bit quantization

### 2.1 Rounding error

The simplest issue is that low-bit representations are coarse. Uniform round-to-nearest is the default baseline, but it wastes capacity when the data distribution is skewed or heavy-tailed.

### 2.2 Outliers

LLM weights and activations often contain outliers. A few large values can force the quantizer scale to grow, which reduces effective resolution for the rest of the tensor.

### 2.3 Layer sensitivity

Not all layers react the same way to noise. Some layers can absorb a lot of quantization error; others are brittle.

### 2.4 Error accumulation

Even small layer-wise errors can compound through depth. A method that looks good on a single matrix may still fail at end-to-end perplexity.

### 2.5 Weight-activation interaction

Weight-only PTQ is easier. Activation quantization is harder because activations vary by input, token position, and layer context.

### 2.6 Hardware and calibration constraints

A good method must work with limited calibration data and must not introduce expensive transforms that erase the gains from compression.

## 3. Representative PTQ families

### 3.1 RTN: the baseline

Round-to-nearest is the simplest PTQ method. It is cheap and stable, but it does not account for sensitivity, outliers, or layer structure.

Use it as a floor, not as a target.

### 3.2 GPTQ: sensitivity-aware weight-only PTQ

GPTQ is one of the most influential modern PTQ methods for LLMs.

Paper:
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

Core idea:
- treat quantization as a reconstruction problem
- use approximate second-order information from calibration data
- correct quantization decisions layer by layer

What it solves:
- weight sensitivity
- per-layer error propagation

What it does not solve directly:
- activation outliers
- runtime transform overhead
- basis misalignment

Why it matters:
- it shows that naive rounding is not enough
- it is a strong weight-only baseline

### 3.3 SmoothQuant: shift difficulty from activations to weights

SmoothQuant is a key activation-aware PTQ method.

Paper:
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)

Core idea:
- smooth activation outliers by moving difficulty into weights via scaling
- make W8A8 feasible without training

What it solves:
- activation outliers
- imbalance between weights and activations

What it teaches:
- scaling is often as important as quantization
- a reparameterization can make quantization easier without changing the function

### 3.4 AWQ: activation-aware weight quantization

AWQ is another major method for LLM PTQ.

Paper:
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

Core idea:
- protect important weights using activation statistics
- use a hardware-friendly strategy for weight-only low-bit quantization

What it solves:
- sensitivity to activation patterns
- weight-only compression at low bit width

What it teaches:
- not every channel contributes equally
- a small subset of “important” channels can dominate accuracy
- hardware-aware simplifications matter

### 3.5 QuaRot: orthogonal rotations for outlier removal

QuaRot is the clearest predecessor for the rotor idea in this repo.

Paper:
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)

Core idea:
- apply orthogonal rotations to hidden states and related projections
- use the invariance of the transformer computation to move into a better basis
- quantize the rotated model end-to-end

What it solves:
- hidden-state outliers
- basis misalignment
- end-to-end 4-bit inference across weights, activations, and KV cache

What it teaches:
- rotation is a valid PTQ primitive
- the point is not mathematical elegance, but outlier reduction
- the method only matters if it improves the full model behavior

### 3.6 ParoQuant: pairwise rotation plus scaling

ParoQuant is the most relevant recent evolution of the rotation idea.

Papers:
- [ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](https://arxiv.org/abs/2511.10645)
- [OpenReview page](https://openreview.net/forum?id=1USeVjsKau)

Core idea:
- use independent pairwise Givens rotations
- combine them with channel-wise scaling
- keep the transform lightweight enough for runtime use

What it solves:
- outliers
- dynamic range within quantization groups
- the overhead problem that often kills more expressive transforms

What it teaches:
- rotation alone is not enough
- scaling and rotation are complementary
- the best PTQ methods are often system-aware, not just accuracy-aware

## 4. What these methods collectively suggest

The literature points to a recurring pattern:

1. Quantization errors are driven by outliers and sensitivity, not just bit width.
2. Basis changes matter.
3. Scaling matters.
4. Calibration objectives matter.
5. Efficiency matters.

This means a strong PTQ method usually combines several ingredients:

- a cheap transform
- a sensitivity-aware objective
- a way to handle outliers
- a hardware-friendly implementation

## 5. Where rotors fit

Rotors are best understood as a structured orthogonal transform family.

They are attractive because:
- they are differentiable
- they are low-parameter compared with dense orthogonal matrices
- they can be applied blockwise or pairwise
- they align well with the rotation-based PTQ literature

They are limited because:
- rotation alone does not directly solve activation outliers
- weight NMSE can improve without perplexity improving
- a pure rotor can be weaker than methods that also scale or clip

This is why the rotor idea should be studied as part of a PTQ pipeline, not as a standalone claim.

## 6. What to compare against in this repo

For a serious study, compare the rotor path against:

- RTN
- GPTQ-style sensitivity-aware weight PTQ
- SmoothQuant-style scaling
- AWQ-style activation-aware weight selection
- QuaRot-style fixed rotation
- ParoQuant-style pairwise rotation plus scaling, if you want a closer modern target

## 7. What the next rotor study should answer

The rotor study should answer these questions:

1. Does the rotor reduce quantization error on the right layers?
2. Does that error reduction predict perplexity?
3. Is the gain still present when scaling or clipping baselines are added?
4. Can the transform stay cheap enough to deploy?
5. Which layers benefit, and which do not?

If the answer is only “it lowers NMSE,” then the rotor is a useful analysis tool but not yet a complete PTQ method.
