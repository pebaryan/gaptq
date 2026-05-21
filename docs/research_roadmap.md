# Research Roadmap

This is the decision document for the PTQ study.

For the literature review, see [literature_review.md](literature_review.md).
For the broader GA study direction, see [ga_research_track.md](ga_research_track.md).
For the execution protocol, see [study_protocol.md](study_protocol.md).
For the executable GA diagnostics, see [gaptq/experimental/ga_analysis.py](../gaptq/experimental/ga_analysis.py).

The core question is not whether quantization reduces model size. That is already obvious. The real question is:

> How do we compress a model while preserving task behavior, under a strict budget for memory, latency, and implementation complexity?

The rotor idea in this repo should be treated as one possible answer to a narrower subproblem:

> Can we learn a cheap orthogonal preconditioner that makes weights or activations easier to quantize?

## 1. What the literature suggests

The review points to five recurring constraints:

1. Rounding is not enough.
2. Outliers matter a lot.
3. Sensitivity varies by layer.
4. Scaling or reparameterization often helps as much as the quantizer itself.
5. Runtime cost matters as much as accuracy.

That is the frame in which rotors should be studied.

## 2. Where rotors fit

Rotors are best viewed as a structured orthogonal preconditioner.

They can help when:
- channel statistics are badly aligned with the quantizer
- a few dimensions dominate the range
- the layer is sensitive to axis-aligned quantization
- a small, cheap basis change can reduce error

They are less compelling when:
- the layer already behaves well under uniform quantization
- scaling or clipping would solve the issue more directly
- the transform is too expensive or too expressive
- the local reconstruction loss does not correlate with perplexity

For a broader GA framing beyond rotation, see [ga_research_track.md](ga_research_track.md).

### Projection-residual status

The current projection-residual branch is now treated as a negative result:

- it can be made task-aware at rank selection time
- it reduces local reconstruction error strongly
- it still does not improve perplexity reliably on `gpt2` or `gpt2-medium`

Keep it as an archived comparison point. Do not spend further study budget on this exact implementation unless the objective changes materially.

### Grade-allocation status

The current live subspace candidate is grade-aware allocation on projection-heavy layers.
The broad `(?:attn|mlp)\.c_proj$` slice improved `gpt2-medium` modestly, but an
attention-only slice regressed. Treat this as the next active PTQ branch, but not yet
as a validated method.

## 3. First research hypotheses for rotors

These are the hypotheses the repo should test:

1. Orthogonal preconditioning reduces outlier concentration.
2. Reduction in weight NMSE does not necessarily imply better perplexity.
3. Some layers benefit from rotation more than others.
4. Rotation is more effective when combined with scaling.
5. A structured low-cost transform can outperform a dense or overfit transform.

## 4. Study design for rotors

### 4.1 Baselines

Compare:
- FP16
- RTN
- fixed orthogonal transform
- learned rotor
- learned rotor plus scaling, if added later

### 4.2 Models

Start with:
- `gpt2`
- `gpt2-medium`

Only expand to larger models once the small-scale study is interpretable.

### 4.3 Metrics

Measure both local and end-to-end quality:
- layer weight NMSE
- layer output MSE on calibration data
- perplexity on a stable validation set
- improvement by layer type
- improvement by depth
- calibration cost and runtime

### 4.4 Rotor-specific ablations

Run the following ablations:
- rotor only vs rotor plus scaling
- per-layer vs grouped layers
- attention-only vs MLP-only vs all layers
- different numbers of optimization steps
- different calibration set sizes
- fixed random rotor vs optimized rotor

### 4.5 Decision criterion

The rotor idea is worth pursuing if it does at least one of the following consistently:
- improves perplexity over RTN on multiple models
- improves calibration loss in a way that predicts perplexity
- combines well with scaling or clipping
- remains cheap enough to deploy

If it only improves NMSE and not task behavior, it is still a useful diagnostic tool, but not yet a strong PTQ method.

## 5. Experiment Matrix

This is the concrete study grid for the next round of work.

### 5.1 Methods to compare

| Method | Description | Primary question |
|---|---|---|
| FP16 | Full precision reference | What is the uncompressed ceiling? |
| RTN | Direct weight quantization | How far does naive quantization go? |
| Fixed orthogonal | Hadamard or random orthogonal basis change | Does a non-learned basis already help? |
| Learned rotor | Blockwise learned orthogonal preconditioning | Can a learned basis change improve PTQ? |
| Learned reflection | Householder reflection preconditioning | Archived comparison point; does a cheaper GA transform help? |
| Learned rotor + scaling | Rotor combined with channel scaling | Does scaling close the gap to stronger PTQ? |
| Grade allocation | Grade-aware bit allocation on projection-heavy layers | Can geometric structure guide precision allocation better than fixed bit maps? |
| Projection + residual model | Projection onto a subspace with explicit residual handling | Can subspace splitting be made task-aware enough to matter? |

### 5.2 Metrics to record

| Metric | What it measures | Why it matters |
|---|---|---|
| FP16 perplexity | Upper-bound reference | Establishes the baseline task quality |
| Quantized perplexity | End-to-end model quality | The main success criterion |
| Layer NMSE | Weight reconstruction error | Useful local proxy, but not sufficient |
| Layer output MSE | Calibration reconstruction quality | Better aligned with task behavior than NMSE |
| Improvement by layer type | Sensitivity by module class | Shows where the method actually helps |
| Improvement by depth | Sensitivity across the stack | Reveals error accumulation patterns |
| Runtime / calibration cost | Practical overhead | Determines whether the method is deployable |

### 5.3 Experimental slices

Run each method on:

| Slice | Purpose |
|---|---|
| `gpt2` | Small-scale sanity check |
| `gpt2-medium` | Larger model sensitivity test |
| Attention-only layers | Isolate attention sensitivity |
| MLP-only layers | Isolate feed-forward sensitivity |
| All layers | End-to-end benchmark |

### 5.4 Decision rules

Treat a method as promising if it satisfies at least one of these:

1. It improves perplexity over RTN on both `gpt2` and `gpt2-medium`.
2. It improves calibration loss in a way that predicts perplexity.
3. It stays cheap enough that the runtime cost does not erase the benefit.
4. It shows a consistent gain on a meaningful subset of layers.

Treat it as exploratory if:

- it improves NMSE but not perplexity
- it only helps on one model size
- it depends on an expensive transform that is hard to deploy
- it is an archived comparison point rather than a live candidate

## 6. Execution Checklist

Use the same evaluation settings for every core run unless a section explicitly says otherwise.

Stable core settings:

```bash
--n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

### 6.1 Baseline pass

Run:

```bash
python -m gaptq.quantize_model --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- FP16 perplexity
- RTN perplexity
- learned rotor perplexity
- mean NMSE gain
- layers improved

Pass condition:
- both commands complete without error
- results are finite

### 6.2 Fixed orthogonal comparison

Run the matrix-level experiment script or an equivalent fixed-rotation path.

For example:

```bash
python -m gaptq.experiment
```

Record:
- NMSE versus RTN
- perplexity if the path is promoted to the GPT-2 benchmark

Pass condition:
- fixed orthogonal transform is at least a meaningful baseline and not obviously worse than RTN on every layer

### 6.3 Learned rotor study

Run the core benchmark again with the learned rotor path.

Record:
- runtime per layer
- per-layer NMSE improvement
- perplexity change versus RTN
- whether gains are concentrated in attention or MLP

Pass condition:
- learned rotor improves RTN on at least one model in perplexity or shows a strong, consistent local gain that justifies further study

### 6.4 Rotor plus scaling study

Only do this if the learned rotor path is promising enough to justify another moving part.

Run:

```bash
python -m gaptq.quantize_model --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

with the scaling variant added in code.

Record:
- whether scaling improves perplexity versus rotor-only
- whether scaling reduces variance across layers
- whether the combined transform remains cheap

Pass condition:
- scaling improves end-to-end quality without making the method too expensive

### 6.4b Projection-residual archive

Do not continue the current projection-residual line as a mainline PTQ method.

What it already showed:
- projection on calibration covariance is locally effective
- task-aware rank selection is better than reconstruction-based selection
- local gains still do not translate into better perplexity

If this line returns, it should be as a new method family with a different objective, not as a continuation of the current branch.

### 6.4c Projection + residual model archive

This branch is now archived as a negative result.

It showed:
- the residual correction can be fit on the calibration data
- the calibration fit can still blow up perplexity
- the extra factorization is too expensive to justify as written

If you revisit this idea, it should be with a materially different residual objective, not the current low-rank correction.

The next live subspace experiment should instead focus on either:
- a more constrained grade-aware allocation scheme, or
- a smaller, task-aware subspace transform with a stronger regularizer

### 6.4d Grade allocation study

This is the current live candidate.

Run the broad projection-heavy slice first:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- whether grade allocation improves perplexity versus RTN
- whether the effect survives scale-up to `gpt2-medium`
- whether the gain is concentrated in `c_proj` layers
- whether an attention-only slice is clearly worse

Pass condition:
- a broad, geometry-aware grade allocation beats RTN on at least one model without
  becoming slower or more brittle than the learned rotor path

### 6.5 Model-size sensitivity

Repeat the chosen best method on:

```bash
python -m gaptq.quantize_model --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- whether the gain survives scale-up
- whether the method gets less stable with depth
- whether certain layer types become more sensitive

Pass condition:
- the method does not collapse on `gpt2-medium`

### 6.6 Layer-slice study

Run attention-only and MLP-only slices if you expose them in code.

Record:
- which layer family is most sensitive
- whether rotors help one family much more than the other

Pass condition:
- the result identifies a specific target region for future work

## 7. How to read the current repo

The current codebase should be read as an exploration of the following question:

> Can a learned orthogonal basis change, parameterized as a rotor, improve PTQ enough to matter?

The current evidence says:
- the idea can improve local quantization error
- the translation to perplexity is mixed
- scaling and task-aware objectives are probably missing pieces

That is a good research starting point.
