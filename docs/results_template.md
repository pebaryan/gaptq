# PTQ Results Template

Use this template to record runs from the study protocol.

## Run Metadata

| Field | Value |
|---|---|
| Date |  |
| Environment |  |
| Device |  |
| Model |  |
| Method |  |
| Core settings |  |

## Core Metrics

| Metric | Value |
|---|---|
| FP16 perplexity |  |
| RTN perplexity |  |
| Quantized perplexity |  |
| Mean NMSE gain vs RTN |  |
| Layers improved |  |
| Runtime |  |

## Layer Notes

| Layer family | Observation |
|---|---|
| Attention |  |
| MLP |  |
| Early layers |  |
| Mid layers |  |
| Late layers |  |

## Interpretation

| Question | Notes |
|---|---|
| Did the method improve perplexity? |  |
| Did NMSE improvement predict perplexity? |  |
| Was the transform cheap enough? |  |
| Would this justify scaling / clipping / more grouping? |  |

## Local `scabi` Notes

Keep any environment-specific shell setup here if needed.
For example, store local-only commands for activating the `scabi` conda env or setting `PYTHONIOENCODING`.

This section is intentionally kept out of the main public narrative. If you want a machine-specific note, put it in [docs/local_scabi.md](local_scabi.md), which is ignored by Git.

