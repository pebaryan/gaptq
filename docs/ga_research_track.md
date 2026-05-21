# Geometric Algebra Research Track for PTQ

This document expands the rotor idea into a broader geometric-algebra study.

The question is not just whether rotation helps. The deeper question is:

> Which geometric structure of a layer should be exposed before quantization?

Rotation is one candidate. GA gives us a language for other candidates too.

## 1. Why go beyond rotation

A rotor is an orthogonal basis change. That is useful, but it only addresses one kind of structure:

- axis misalignment
- directional outliers
- channel imbalance

PTQ error can also come from:

- subspace mismatch
- redundant directions that should be suppressed
- components that should be projected out rather than rotated
- grade-specific structure that may not be equally quantization-sensitive

If the layer’s geometry is not mainly rotational, a rotor can only do so much.

## 2. GA concepts worth studying for PTQ

### 2.1 Reflections and versors

Rotors are composed from reflections. That suggests a richer family of transforms:

- single reflections
- products of reflections
- constrained versors that may be more local than a full rotor block

The current reflection baseline is now archived as a negative result:
- it was cheap
- it produced small local NMSE changes
- it did not improve perplexity on `gpt2` or `gpt2-medium`

So reflection is useful as a sanity check, but it is not the next main PTQ direction.

### 2.2 Grade decomposition

GA naturally separates multivector content into grades.

Potential PTQ use:

- analyze whether different grades are more quantization-sensitive
- see whether some geometric components dominate outliers
- test whether grade-aware precision allocation helps

This is more general than the earlier per-grade experiment, because the goal is analysis first, compression second.

### 2.3 Projections and residuals

GA has a clean notion of projection onto a subspace.

Potential PTQ use:

- project a tensor into a quantization-friendly subspace
- quantize the projected part aggressively
- store or approximate the residual separately
- learn a residual model that compensates for what the projection throws away

This is conceptually closer to compression than rotation alone.

### 2.4 Subspace-specific transforms

Instead of one transform for the whole layer, use transforms for:

- attention heads
- MLP groups
- row/column blocks
- identified outlier subspaces

Potential PTQ use:

- better match model structure
- avoid forcing the same transform on layers that behave differently

## 3. Hypotheses for GA beyond rotation

These are the hypotheses worth testing next:

1. Some layers are more projection-like than rotation-like.
2. Grade structure correlates with quantization sensitivity.
3. A small number of reflections may be enough to expose a quantization-friendly basis.
4. Grouped transforms will outperform global transforms on sensitive layers.
5. A transform that removes or isolates residual structure may beat a pure orthogonal map.

## 4. Recommended study order

Do not start with a more complicated transform.

Start with analysis:

1. Measure outlier concentration by layer.
2. Measure grade-like structure or spectral decomposition signals.
3. Compare rotor, projection-based, and residual-aware variants on the same calibration set.

Then move to compression:

1. Rotor only.
2. Rotor plus scaling.
3. Grade-aware bit allocation on the projection-heavy layers.
4. Projection plus explicit residual modeling only if the objective changes materially.

## 4.1 First executable checklist

Run the following diagnostics before changing the transform again.

### Step A: Weight geometry

Compute these layer stats for `gpt2` and `gpt2-medium`:

- global outlier ratio: `max(abs(W)) / rms(W)`
- row outlier ratio
- column outlier ratio
- singular-value entropy
- singular-value dynamic range
- RTN NMSE
- grade fractions for square layers

Command:

```bash
python -m gaptq.experimental.ga_analysis --model gpt2 --n-bits 4 --eval-batches 10 --batch-size 2 --max-length 64 --json-out docs/ga_gpt2.json
python -m gaptq.experimental.ga_analysis --model gpt2-medium --n-bits 4 --eval-batches 10 --batch-size 2 --max-length 64 --json-out docs/ga_gpt2_medium.json
```

### Step B: Activation geometry

Collect activation spectra at the transformer-block level:

- input spectral entropy
- output spectral entropy
- input dynamic range
- output dynamic range

Command:

```bash
python -m gaptq.experimental.ga_analysis --model gpt2 --n-bits 4 --eval-batches 10 --batch-size 2 --max-length 64 --with-activations
```

### Step C: Compare against rotor behavior

Use the same layers and look for correlations between:

- high outlier ratio and rotor gain
- high singular-value dynamic range and rotor gain
- grade imbalance and rotor gain
- activation dynamic range and rotor gain

If no correlation appears, then the rotor is probably not the right transform family to expand first.

### Initial observation from the first smoke runs

On `gpt2-medium`, the largest geometry and RTN-error hotspots were the MLP projection layers
(`mlp.c_proj`) and some attention projections. That matches the intuition that a few
projection layers dominate the low-bit problem.

At the same time, the one-step rotor diagnostic did not move far enough from the zero-angle
initialization to produce meaningful layer-wise gains on `gpt2-medium`. That suggests the
current rotor optimizer is too weak for the larger model, or that the rotor-only objective is
the wrong thing to optimize on medium-scale layers.

This is useful because it narrows the next question:

> Is the failure caused by the transform family, or by the optimization objective and budget?

The next experiments should answer that before we add more geometric expressiveness.

### Projection-residual status

The current projection-residual implementation is archived as a negative result.

It did validate one useful point:
- activation-covariance bases can strongly reduce local reconstruction error
- task-aware rank selection is better than pure reconstruction-based selection

But it did not validate the method as an end-to-end PTQ direction:
- perplexity did not improve reliably
- the branch remained too local even with held-out task loss

That means the next GA work should move toward either:
- a more constrained grade-aware allocation scheme, or
- a smaller, task-aware subspace transform with a stronger regularizer

### Grade-allocation status

The current live candidate is grade-aware allocation on projection-heavy layers.
The broad `(?:attn|mlp)\.c_proj$` slice is the most promising one so far. An
attention-only slice regressed, so this should be treated as a geometry-guided
allocation study, not as a blanket replacement for rotor-only quantization.

## 5. What to look for in the data

For each candidate transform, measure:

- layer weight NMSE
- calibration output MSE
- perplexity
- outlier reduction
- runtime / overhead
- whether the effect is concentrated in attention or MLP blocks

The key question is whether the GA structure predicts a better quantization basis.

## 6. What would count as a useful result

The GA direction is useful if it shows one of these:

- a projection-plus-residual variant improves over rotor-only
- grade structure predicts where precision should go
- some layers respond better to suppression than rotation
- the GA view identifies a cheaper transform with the same or better quality

If none of those hold, then GA is still useful as analysis language, but not as the main PTQ mechanism.
