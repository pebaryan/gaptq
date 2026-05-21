"""Geometric-algebra analysis for PTQ.

This script is intentionally diagnostic, not a quantizer. It computes the
first-order statistics that should tell us whether a GA-based transform is
worth pursuing beyond rotation:

- weight outlier concentration
- singular-value entropy / dynamic range
- grade decomposition balance for square layers
- RTN reconstruction error as a local sensitivity proxy
- optional activation spectra by transformer block
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..ga import grade_decompose
from ..quantization import UniformQuantizer, quantization_error
from ..quantize_model import (
    clone_model,
    evaluate_perplexity,
    get_linear_layers,
    get_weight,
    load_model_and_tokenizer,
)
from .diagnose_rotor import collect_activation_spectra


def _spectral_stats(W: torch.Tensor) -> Dict[str, float]:
    """Compute singular-value entropy and dynamic range."""
    if W.numel() == 0:
        return {
            "sv_entropy": float("nan"),
            "sv_kurtosis": float("nan"),
            "sv_dynamic_range_db": float("nan"),
        }

    s = torch.linalg.svdvals(W.float())
    if s.numel() == 0:
        return {
            "sv_entropy": float("nan"),
            "sv_kurtosis": float("nan"),
            "sv_dynamic_range_db": float("nan"),
        }

    p = (s / s.sum().clamp(min=1e-12)).clamp(min=1e-12)
    entropy = -(p * torch.log(p)).sum().item()
    if s.numel() < 4:
        kurtosis = 0.0
    else:
        mu = s.mean()
        var = ((s - mu) ** 2).mean().clamp(min=1e-12)
        kurtosis = ((s - mu) ** 4).mean().div(var ** 2).item()
    dynamic_range = (
        20 * math.log10((s[0] / s[-1].clamp(min=1e-12)).item())
        if s.numel() > 1 else 0.0
    )
    return {
        "sv_entropy": entropy,
        "sv_kurtosis": kurtosis,
        "sv_dynamic_range_db": dynamic_range,
    }


def _grade_stats(W: torch.Tensor) -> Dict[str, float]:
    """Compute grade decomposition stats for square matrices."""
    if W.shape[0] != W.shape[1] or W.numel() == 0:
        return {
            "grade_scalar_frac": float("nan"),
            "grade_bivector_frac": float("nan"),
            "grade_strain_frac": float("nan"),
        }

    scalar, bivector, strain = grade_decompose(W)
    total = (W.pow(2).sum() + 1e-12)
    return {
        "grade_scalar_frac": scalar.pow(2).sum().div(total).item(),
        "grade_bivector_frac": bivector.pow(2).sum().div(total).item(),
        "grade_strain_frac": strain.pow(2).sum().div(total).item(),
    }


def analyze_layer(
    name: str,
    module: torch.nn.Module,
    layer_type: str,
    n_bits: int = 4,
) -> Dict[str, float]:
    """Analyze one linear/Conv1D layer."""
    W = get_weight(module, layer_type)
    q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
    W_rtn = q(W)

    rms = W.pow(2).mean().sqrt().item()
    max_abs = W.abs().max().item()
    outlier_ratio = max_abs / (rms + 1e-12)
    row_rms = W.pow(2).mean(dim=1).sqrt()
    row_max = W.abs().amax(dim=1)
    row_outlier_ratio = (row_max / (row_rms + 1e-12)).mean().item()
    col_rms = W.pow(2).mean(dim=0).sqrt()
    col_max = W.abs().amax(dim=0)
    col_outlier_ratio = (col_max / (col_rms + 1e-12)).mean().item()

    spectral = _spectral_stats(W)
    grades = _grade_stats(W)
    nmse_rtn = quantization_error(W, W_rtn).item()

    stats = {
        "layer": name,
        "type": layer_type,
        "shape_rows": float(W.shape[0]),
        "shape_cols": float(W.shape[1]),
        "is_square": float(int(W.shape[0] == W.shape[1])),
        "rms": rms,
        "max_abs": max_abs,
        "outlier_ratio": outlier_ratio,
        "row_outlier_ratio": row_outlier_ratio,
        "col_outlier_ratio": col_outlier_ratio,
        "nmse_rtn": nmse_rtn,
        **spectral,
        **grades,
    }
    return stats


def _print_ranked(rows: List[Dict[str, float]], key: str, title: str, top_k: int = 10) -> None:
    ranked = sorted(rows, key=lambda x: x.get(key, float("-inf")), reverse=True)[:top_k]
    print(f"\n{title}")
    for row in ranked:
        layer = row["layer"]
        value = row.get(key, float("nan"))
        print(f"  {layer:40s} {value:10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GA diagnostics for PTQ layers")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--n-bits", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--with-activations", action="store_true",
                        help="Also collect activation spectra by transformer block")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to write results as JSON")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, device = load_model_and_tokenizer(args.model, device)
    layers = get_linear_layers(model)

    print("=" * 72)
    print(f"GA ANALYSIS: {args.model}")
    print("=" * 72)

    fp16 = clone_model(model).half().to(device)
    ppl_fp16 = evaluate_perplexity(
        fp16, tokenizer,
        num_batches=args.eval_batches,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    del fp16
    print(f"FP16 perplexity: {ppl_fp16:.2f}")

    rows = []
    for name, module, layer_type in layers:
        stats = analyze_layer(name, module, layer_type, n_bits=args.n_bits)
        rows.append(stats)

    print(f"Analyzed {len(rows)} layers")
    _print_ranked(rows, "outlier_ratio", "Top layers by global outlier ratio", args.top_k)
    _print_ranked(rows, "nmse_rtn", "Top layers by RTN NMSE", args.top_k)
    _print_ranked(rows, "sv_dynamic_range_db", "Top layers by singular-value dynamic range", args.top_k)

    square_rows = [r for r in rows if r["is_square"] > 0.5]
    if square_rows:
        _print_ranked(square_rows, "grade_bivector_frac", "Top square layers by bivector fraction", args.top_k)

    activation_stats = {}
    if args.with_activations:
        activation_stats = collect_activation_spectra(
            model, tokenizer,
            n_batches=max(1, args.eval_batches // 5),
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        print(f"\nCollected activation spectra for {len(activation_stats)} blocks")
        for block_idx in sorted(activation_stats.keys())[:args.top_k]:
            s = activation_stats[block_idx]
            print(
                f"  block {block_idx:2d}: "
                f"in_entropy={s['entropy_in']:.3f}, "
                f"out_entropy={s['entropy_out']:.3f}, "
                f"in_DR={s['dynamic_range_in_db']:.1f}dB, "
                f"out_DR={s['dynamic_range_out_db']:.1f}dB"
            )

    payload = {
        "model": args.model,
        "n_bits": args.n_bits,
        "fp16_ppl": ppl_fp16,
        "layers": rows,
        "activations": activation_stats,
    }

    if args.json_out:
        path = Path(args.json_out)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
