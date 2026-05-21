# PTQ Study Protocol

This document turns the literature review and roadmap into an executable study procedure.

Use it when you are actually running experiments and recording results.

For background, see:
- [literature_review.md](literature_review.md)
- [external_ptq_comparison.md](external_ptq_comparison.md)
- [research_roadmap.md](research_roadmap.md)
- [results_template.md](results_template.md)
- [results_snapshot.md](results_snapshot.md)

## 1. Study Goal

The study asks whether a learned orthogonal basis change, parameterized as a rotor, helps PTQ enough to matter in end-to-end language modeling.

The key question is not whether the rotor lowers local error. The question is whether it improves task behavior after quantization.

The current projection-residual branch is not part of the active path. It is kept as an archived negative result because it can reduce local error without improving perplexity reliably.

## 2. Fixed Settings

Use these settings for the core runs unless a subsection says otherwise:

```bash
--n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Run everything from the same environment, with the same model checkpoint, tokenizer, and validation corpus.
If you need machine-specific setup notes for `scabi`, keep them in [local_scabi.md](local_scabi.md). That file is ignored by Git and stays local.

## 3. Run Order

### 3.1 Baseline first

Run:

```bash
python -m gaptq.quantize_model --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- FP16 perplexity
- RTN perplexity
- learned rotor perplexity
- mean NMSE gain over RTN
- layers improved

### 3.2 Fixed orthogonal comparison

Run the matrix-level experiment path:

```bash
python -m gaptq.experiment
```

Record:
- RTN NMSE
- fixed orthogonal NMSE
- learned rotor NMSE
- any sign of a gap between fixed and learned transforms

### 3.3 Learned rotor analysis

Use the core benchmark output and inspect:
- attention versus MLP behavior
- early versus late layers
- layers with positive versus negative improvement

Record:
- runtime per layer
- per-layer NMSE change
- perplexity change
- whether rotor gains are concentrated in one module family

### 3.4 Grade allocation

This is the current live subspace candidate.

Run the `mlp.c_proj` slice first:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- whether grade allocation improves perplexity versus RTN
- whether the gain survives on `gpt2-medium`
- whether the effect is concentrated in `c_proj` layers
- whether an attention-only slice is clearly worse

Then optionally run the broad slice as a comparison point:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex '(?:attn|mlp)\.c_proj$' --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex '(?:attn|mlp)\.c_proj$' --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Then run the attention-only slice as a rejection check:

```bash
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex 'attn\.c_proj$' --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --grade-alloc --grade-alloc-regex 'attn\.c_proj$' --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

### 3.5 Reflection baseline

Run the Householder reflection baseline after the rotor pass:

```bash
python -m gaptq.quantize_model --experimental --reflection --model gpt2 --n-bits 4 --n-steps 1 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --reflection --model gpt2-medium --n-bits 4 --n-steps 1 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- whether reflection improves perplexity versus RTN
- whether it beats rotor on the same layers
- whether it is cheaper or more stable than the rotor path

### 3.6 Projection + residual model archive

The explicit residual-model branch is now archived as a negative result.

It showed that:
- the low-rank residual can fit the calibration split
- that fit can still overfit badly
- the extra factorization is expensive relative to the rotor path

Do not continue this branch as written unless the residual objective changes materially.

### 3.7 Rotor plus scaling

Only run this after the learned rotor has shown enough promise to justify another moving part.

Run:

```bash
python -m gaptq.quantize_model --experimental --rotor-scale --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --rotor-scale --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- whether scaling improves perplexity
- whether scaling reduces variance across layers
- whether the extra complexity is justified

### 3.8 Model-size sensitivity

Repeat the chosen best method on `gpt2-medium`.

Record:
- whether the gain survives scale-up
- whether larger depth weakens the method
- whether the method stays stable on harder layers

### 3.9 Slice study

Run attention-only and MLP-only variants if exposed in code.

Record:
- which family benefits most
- which family is sensitive to rotation
- whether a targeted transform would be better than a global one

### 3.10 AWQ-style proxy baseline

This is the strongest current non-GA comparator in-tree.

Run:

```bash
python -m gaptq.quantize_model --experimental --awq-proxy --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --experimental --awq-proxy --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Record:
- whether the proxy beats RTN
- whether it beats the live grade-allocation branch
- whether it remains cheap enough to be a meaningful comparator

## 4. Decision Rules

Treat the rotor direction as promising if it does at least one of these:

1. Improves perplexity over RTN on both `gpt2` and `gpt2-medium`.
2. Improves calibration loss in a way that predicts perplexity.
3. Stays cheap enough that runtime cost does not erase the gain.
4. Reveals a stable pattern that suggests a better structured transform.

Treat it as exploratory if:
- it only improves NMSE
- it only helps on one model size
- it depends on an expensive transform

## 5. Results Template

Use [results_template.md](results_template.md) to record each run.

## 6. Interpretation Rules

Do not treat NMSE improvement as a success criterion by itself.

The rotor idea only becomes a serious PTQ result if the benchmark shows at least one of:
- lower perplexity
- more robust behavior across models
- a clear pathway to a better hybrid method, such as rotor plus scaling

Do not continue the existing projection-residual implementation unless you change the objective or the transform family in a material way.

## 7. Next benchmark plan

The next benchmark should answer whether grade allocation is genuinely helping,
or whether it is just a better heuristic than RTN.

### 7.1 AWQ-style proxy baseline

The repo now has an AWQ-style proxy baseline on `mlp.c_proj`. It is the
strongest current non-GA comparator in-tree and should be treated as the bar
for the live grade-allocation branch.

What the current result says:
- the AWQ proxy beats RTN on both `gpt2` and `gpt2-medium`
- grade allocation still beats the AWQ proxy on both models
- the live question is therefore not whether a stronger baseline exists, but
  whether grade allocation can continue to beat it on harder slices or larger
  models

Why this baseline matters:
- it is a strong practical PTQ comparator
- it targets the same outlier / sensitivity problem without GA decomposition
- it tells us whether the current grade-allocation gain is structural or mostly
  a search heuristic

Current diagonal-scaling comparator:
- improved `gpt2`
- hurt `gpt2-medium`
- keep it as a reference baseline, not a replacement for grade allocation

### 7.2 Run order

1. RTN
2. learned rotor
3. `mlp.c_proj` grade allocation
4. AWQ-style proxy baseline
5. rotor plus scaling, only if needed as a hybrid comparison

### 7.3 Decision rules

Treat grade allocation as promising if:
- it beats RTN on both `gpt2` and `gpt2-medium`
- it stays competitive with the AWQ-style proxy baseline
- it remains cheap enough that runtime overhead is minor

Treat it as a heuristic gain if:
- it beats RTN but loses to the AWQ-style proxy baseline
- it only works on `mlp.c_proj`
- it depends heavily on the current candidate search

Treat it as rejected if:
- the AWQ-style proxy baseline matches or beats it on both models
- the gain disappears when the search is simplified
