# GAP-TQ

GAP-TQ is a research sandbox for one idea: learn an orthogonal preconditioner, parameterized as a geometric-algebra rotor, before post-training quantization.

The useful part of the project is the learned rotation. The geometric-algebra notation is a parameterization and analysis tool, not the goal by itself.

For the research framing, see [docs/research_roadmap.md](docs/research_roadmap.md).
For the literature review, see [docs/literature_review.md](docs/literature_review.md).
For GA beyond rotation, see [docs/ga_research_track.md](docs/ga_research_track.md).

## What the core path does

- Load GPT-2 or GPT-2-medium
- Measure an FP16 baseline
- Quantize weights with RTN
- Quantize weights with a learned block-diagonal rotor
- Optionally test QuaRot-style absorption as a comparison point

## What lives in the repo

- [`gaptq/ga.py`](gaptq/ga.py): rotor and geometric algebra primitives
- [`gaptq/quantization.py`](gaptq/quantization.py): uniform quantization helpers
- [`gaptq/rotor_quant.py`](gaptq/rotor_quant.py): learned rotation + quantization
- [`gaptq/quantize_model.py`](gaptq/quantize_model.py): GPT-2 benchmark runner
- [`gaptq/experiment.py`](gaptq/experiment.py): small matrix-level experiments
- [`gaptq/experimental/`](gaptq/experimental/): side branches that are not part of the core benchmark story

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests -v
python -m gaptq.quantize_model --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1
```

## Current interpretation

The repo currently supports a narrow claim:

- learned orthogonal preconditioning can reduce layer-wise quantization error
- whether that translates to lower perplexity depends on the layer and the model

The experimental branches in `gaptq/experimental/` are exploratory and should be treated separately from the main PTQ path.
The projection-residual experiment is now treated as an archived negative result: it can reduce local error, but it does not reliably improve perplexity.
The reflection baseline is also archived as a negative result: it is cheap and locally stable, but it does not improve perplexity on the benchmark models.
The next live subspace experiment is no longer the low-rank residual model. The data show that branch is a dead end as written.
Grade allocation on `mlp.c_proj` is the current live subspace candidate. The broader `(?:attn|mlp)\.c_proj$` slice was superseded by the MLP-only slice, and the attention-only slice was rejected.

## Benchmark Snapshot

Stable GPT-2 benchmark settings used for the core path:

```bash
python -m gaptq.quantize_model --model gpt2 --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
python -m gaptq.quantize_model --model gpt2-medium --n-bits 4 --n-steps 2 --n-restarts 1 --eval-batches 50 --batch-size 2 --max-length 64
```

Results from the cleaned core benchmark:

| Model | FP16 PPL | RTN PPL | Learned Rotor PPL | Mean NMSE Gain vs RTN | Layers Improved |
|---|---:|---:|---:|---:|---:|
| `gpt2` | 61.76 | 94.93 | 87.56 | 4.6% | 46/48 |
| `gpt2-medium` | 44.17 | 55.19 | 55.16 | 4.4% | 89/96 |

## Limitations

- NMSE improvements do not reliably translate into perplexity improvements.
- The projection-residual branch is archived unless the objective changes materially.
- The projection + residual model is archived unless the residual objective changes materially.
- The Householder reflection branch is archived unless a new objective makes it worthwhile.
- The experimental branches are not part of the core claim and should not be read as validated methods.
- Activation quantization, per-grade quantization, and ensemble quantization remain exploratory.
- Grade allocation is the current live subspace candidate, but the evidence is still mixed and model-dependent even after narrowing to `mlp.c_proj`.
