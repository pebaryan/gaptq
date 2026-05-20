# PTQ Study Protocol

This document turns the literature review and roadmap into an executable study procedure.

Use it when you are actually running experiments and recording results.

For background, see:
- [literature_review.md](literature_review.md)
- [research_roadmap.md](research_roadmap.md)
- [results_template.md](results_template.md)
- [results_snapshot.md](results_snapshot.md)

## 1. Study Goal

The study asks whether a learned orthogonal basis change, parameterized as a rotor, helps PTQ enough to matter in end-to-end language modeling.

The key question is not whether the rotor lowers local error. The question is whether it improves task behavior after quantization.

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

### 3.4 Rotor plus scaling

Only run this after the learned rotor has shown enough promise to justify another moving part.

Record:
- whether scaling improves perplexity
- whether scaling reduces variance across layers
- whether the extra complexity is justified

### 3.5 Model-size sensitivity

Repeat the chosen best method on `gpt2-medium`.

Record:
- whether the gain survives scale-up
- whether larger depth weakens the method
- whether the method stays stable on harder layers

### 3.6 Slice study

Run attention-only and MLP-only variants if exposed in code.

Record:
- which family benefits most
- which family is sensitive to rotation
- whether a targeted transform would be better than a global one

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
