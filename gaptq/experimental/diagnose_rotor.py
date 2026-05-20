"""
Diagnostic: Why does rotor help GPT-2 Small but hurt GPT-2 Medium?

Tests four hypotheses:
  H1: Gradient scale — rotor angles converge differently in larger model
  H2: Activation spectrum shift — rotation affects activation SVD differently
  H3: Error compounding — more layers amplify structured rotor errors
  H4: Layer-depth dependence — rotor efficacy varies by block position

Runs on both models, comparing:
  1. Per-layer NMSE improvement by type and depth
  2. Rotor angle magnitudes
  3. SVD spectrum change of weights after rotation
  4. Activation singular value distribution at each block
  5. FP16/RTN/Rotor PPL comparison
"""

import argparse
import math
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..quantization import UniformQuantizer, quantization_error
from ..ga import block_diag_rotor_matrix
from ..quantize_model import (
    load_model_and_tokenizer, get_linear_layers, get_weight, set_weight,
    clone_model, evaluate_perplexity,
)


def diagnose_layer(
    W: torch.Tensor,
    layer_name: str,
    layer_type: str,
    n_bits: int = 4,
    n_steps: int = 30,
    device: torch.device = None,
) -> Dict:
    """Run rotor optimization on one layer and collect diagnostics."""
    if device is None:
        device = W.device
    dim = W.shape[1]
    n_rotors = max(dim // 2, 1)

    q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
    W_rtn = q(W)
    nmse_rtn = quantization_error(W, W_rtn).item()

    # Run optimization (30 steps, lightweight block-diagonal rotor)
    angles = torch.nn.Parameter(torch.zeros(n_rotors, device=device))
    optimizer = torch.optim.Adam([angles], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    best_angles = None
    best_loss = float("inf")
    loss_trajectory = []

    for step in range(n_steps):
        optimizer.zero_grad()
        R = block_diag_rotor_matrix(angles, dim)
        W_rot = W @ R.T
        W_q = q(W_rot)
        W_deq = W_q @ R
        loss = F.mse_loss(W_deq, W)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([angles], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        loss_trajectory.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_angles = angles.detach().clone()

    # Final evaluation with best angles
    with torch.no_grad():
        angles.copy_(best_angles)
        R_final = block_diag_rotor_matrix(angles, dim)
        W_rot = W @ R_final.T
        W_q_final = q(W_rot) @ R_final

        nmse_rotor = quantization_error(W, W_q_final).item()
        improvement_pct = (nmse_rtn - nmse_rotor) / nmse_rtn * 100
        angle_mag = angles.abs().mean().item()
        angle_max = angles.abs().max().item()

        # SVD spectrum comparison: original vs rotated weights
        _, S_orig, _ = torch.svd(W.float())
        _, S_rot, _ = torch.svd((W @ R_final.T).float())
        # KL divergence between normalized singular value distributions
        p_orig = (S_orig / S_orig.sum()).clamp(min=1e-10)
        p_rot = (S_rot / S_rot.sum()).clamp(min=1e-10)
        svd_kl = (p_orig * (p_orig.log() - p_rot.log())).sum().item()

    return {
        "layer": layer_name,
        "type": layer_type,
        "dim": dim,
        "nmse_rtn": nmse_rtn,
        "nmse_rotor": nmse_rotor,
        "improvement_pct": improvement_pct,
        "angle_mean": angle_mag,
        "angle_max": angle_max,
        "svd_kl": svd_kl,
        "init_loss": loss_trajectory[0] if loss_trajectory else 0,
        "final_loss": loss_trajectory[-1] if loss_trajectory else 0,
    }


@torch.no_grad()
def collect_activation_spectra(
    model: torch.nn.Module,
    tokenizer,
    n_batches: int = 4,
    batch_size: int = 2,
    max_length: int = 128,
    device: torch.device = None,
) -> Dict:
    """Run forward pass and collect activation spectra at each block."""
    if device is None:
        device = next(model.parameters()).device

    n_blocks = len(model.transformer.h)
    activations = {}
    handles = []

    def make_hook(block_idx):
        def hook(module, input, output):
            if block_idx not in activations:
                activations[block_idx] = {}
            activations[block_idx]["h_in"] = input[0].detach().cpu()
            activations[block_idx]["h_out"] = output[0].detach().cpu()
        return hook

    for i in range(n_blocks):
        handle = model.transformer.h[i].register_forward_hook(make_hook(i))
        handles.append(handle)

    # Generate calibration data
    from datasets import load_dataset
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        for h in handles:
            h.remove()
        return {}

    texts = [t for t in dataset["text"] if len(t.strip()) > 20]

    for start_idx in range(0, min(len(texts), n_batches * batch_size), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        if not batch_texts:
            break
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_length,
        ).to(device)
        model(**encoded)

    for h in handles:
        h.remove()

    result = {}
    for block_idx in sorted(activations.keys()):
        h_in = activations[block_idx]["h_in"]
        h_out = activations[block_idx]["h_out"]
        B, S, D = h_in.shape
        h_in_2d = h_in.reshape(-1, D)
        h_out_2d = h_out.reshape(-1, D)

        _, S_in, _ = torch.svd(h_in_2d.float())
        _, S_out, _ = torch.svd(h_out_2d.float())

        def spectral_entropy(S):
            p = S / S.sum()
            return -(p * torch.log(p + 1e-10)).sum().item()

        def spectral_kurtosis(S):
            if S.numel() < 4:
                return 0.0
            mu = S.mean()
            var = ((S - mu) ** 2).mean()
            return ((S - mu) ** 4).mean() / (var ** 2 + 1e-10)

        result[block_idx] = {
            "entropy_in": spectral_entropy(S_in),
            "entropy_out": spectral_entropy(S_out),
            "kurtosis_in": spectral_kurtosis(S_in),
            "kurtosis_out": spectral_kurtosis(S_out),
            "dynamic_range_in_db": (20 * math.log10((S_in[0] / (S_in[-1] + 1e-10)).item())) if len(S_in) > 1 else 0,
            "dynamic_range_out_db": (20 * math.log10((S_out[0] / (S_out[-1] + 1e-10)).item())) if len(S_out) > 1 else 0,
        }

    return result


def diagnose_model(
    model_name: str,
    n_bits: int = 4,
    n_steps: int = 30,
    n_eval_batches: int = 10,
    batch_size: int = 2,
    max_length: int = 64,
    device: torch.device = None,
) -> Dict:
    """Full diagnostic for one model."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model_and_tokenizer(model_name, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_blocks = len(model.transformer.h)
    hidden_dim = model.transformer.h[0].attn.embed_dim
    layers = get_linear_layers(model)
    print(f"Params: {n_params:,}, Blocks: {n_blocks}, Hidden dim: {hidden_dim}, Linear layers: {len(layers)}")

    results = {
        "model_name": model_name,
        "n_params": n_params,
        "n_blocks": n_blocks,
        "hidden_dim": hidden_dim,
        "n_layers": len(layers),
    }

    # ── 1. Per-layer diagnostics ──
    print(f"\n─── Layer-by-layer rotor diagnostics ───")
    per_layer = []
    for i, (name, mod, ltype) in enumerate(layers):
        W = get_weight(mod, ltype)

        print(f"  [{i+1}/{len(layers)}] {name}: {list(W.shape)} — running diag...", end="", flush=True)
        diag = diagnose_layer(W, name, ltype, n_bits=n_bits, n_steps=n_steps, device=device)
        print(f"\r  [{i+1}/{len(layers)}] {name:55s} "
              f"θ={diag['angle_mean']:.4f} rad, "
              f"NMSE: {diag['nmse_rtn']:.4f}→{diag['nmse_rotor']:.4f} "
              f"({diag['improvement_pct']:+.1f}%)")
        per_layer.append(diag)
    results["per_layer"] = per_layer

    # ── Aggregate by layer type ──
    by_type = {}
    for d in per_layer:
        lt = d["type"]
        by_type.setdefault(lt, []).append(d["improvement_pct"])
    print(f"\n─── Improvement by layer type ───")
    for lt, imps in sorted(by_type.items()):
        mean = sum(imps) / len(imps)
        n_pos = sum(1 for i in imps if i > 0)
        print(f"  {lt:10s}: mean {mean:+.1f}%, {n_pos}/{len(imps)} improved")
    results["by_type"] = {lt: {"mean": sum(imps)/len(imps), "n_improved": sum(1 for i in imps if i > 0), "total": len(imps)} for lt, imps in by_type.items()}

    # ── Aggregate by block depth ──
    by_depth = {"early": [], "mid": [], "late": []}
    for d in per_layer:
        m = re.match(r"transformer\.h\.(\d+)", d["layer"])
        if m:
            block_idx = int(m.group(1))
            if block_idx < n_blocks / 3:
                by_depth["early"].append(d["improvement_pct"])
            elif block_idx < 2 * n_blocks / 3:
                by_depth["mid"].append(d["improvement_pct"])
            else:
                by_depth["late"].append(d["improvement_pct"])
    print(f"\n─── Improvement by block depth ───")
    for depth, imps in by_depth.items():
        if imps:
            print(f"  {depth:10s}: mean {sum(imps)/len(imps):+.1f}%, {len(imps)} layers")
    results["by_depth"] = {d: {"mean": sum(imps)/len(imps) if imps else 0, "n": len(imps)} for d, imps in by_depth.items() if imps}

    # ── 2. PPL comparison ──
    print(f"\n─── Perplexity comparison ───")

    model_fp16 = clone_model(model).half().to(device)
    t0 = time.time()
    ppl_fp16 = evaluate_perplexity(model_fp16, tokenizer, num_batches=n_eval_batches, batch_size=batch_size, max_length=max_length, device=device)
    print(f"  FP16:           {ppl_fp16:.2f} PPL (in {time.time()-t0:.1f}s)")
    del model_fp16

    # RTN
    model_rtn = clone_model(model).to(device)
    q_rtn = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
    for name, mod, ltype in layers:
        W = get_weight(mod, ltype)
        set_weight(mod, ltype, q_rtn(W))
    ppl_rtn = evaluate_perplexity(model_rtn, tokenizer, num_batches=n_eval_batches, batch_size=batch_size, max_length=max_length, device=device)
    print(f"  RTN (4-bit):    {ppl_rtn:.2f} PPL (+{(ppl_rtn/ppl_fp16-1)*100:+.1f}%)")
    del model_rtn

    # Rotor (all layers, using get_optimal_rotation)
    from ..rotor_quant import get_optimal_rotation
    model_rotor = clone_model(model).to(device)
    q_rot = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
    n_rotor_layers = 0
    for name, mod, ltype in layers:
        W = get_weight(mod, ltype)
        R = get_optimal_rotation(W, mode="block", n_bits=n_bits, n_optimization_steps=n_steps, verbose=False)
        W_rot = W @ R.T
        W_q = q_rot(W_rot) @ R
        set_weight(mod, ltype, W_q)
        n_rotor_layers += 1
    ppl_rotor = evaluate_perplexity(model_rotor, tokenizer, num_batches=n_eval_batches, batch_size=batch_size, max_length=max_length, device=device)
    print(f"  Rotor (4-bit):  {ppl_rotor:.2f} PPL (+{(ppl_rotor/ppl_fp16-1)*100:+.1f}%) [{n_rotor_layers} layers]")
    del model_rotor

    results["ppl"] = {"fp16": ppl_fp16, "rtn": ppl_rtn, "rotor": ppl_rotor}

    # ── 3. Activation spectra ──
    print(f"\n─── Activation spectra ───")
    spectra = collect_activation_spectra(model, tokenizer, n_batches=4, batch_size=2, max_length=max_length, device=device)
    print(f"  Captured {len(spectra)} blocks")
    for bidx in sorted(list(spectra.keys())[:5]):
        s = spectra[bidx]
        print(f"  Block {bidx:2d}: in_entropy={s['entropy_in']:.2f}, out_entropy={s['entropy_out']:.2f}, in_DR={s['dynamic_range_in_db']:.0f}dB")
    results["spectra"] = spectra

    # ── Summary ──
    impr_pcts = [d["improvement_pct"] for d in per_layer]
    angle_means = [d["angle_mean"] for d in per_layer]
    print(f"\n─── SUMMARY ───")
    if impr_pcts:
        print(f"  Mean NMSE improvement: {sum(impr_pcts)/len(impr_pcts):+.1f}%")
        print(f"  Layers improved:       {sum(1 for i in impr_pcts if i > 0)}/{len(impr_pcts)}")
    if angle_means:
        print(f"  Mean angle magnitude:  {sum(angle_means)/len(angle_means):.4f} rad")
    print(f"  PPL: FP16={ppl_fp16:.2f}, RTN={ppl_rtn:.2f}, Rotor={ppl_rotor:.2f}")
    print(f"  Rotor ΔPPL vs RTN:    {ppl_rotor - ppl_rtn:+.2f}")

    results["summary"] = {
        "mean_improvement": sum(impr_pcts) / len(impr_pcts) if impr_pcts else 0,
        "n_improved": sum(1 for i in impr_pcts if i > 0) if impr_pcts else 0,
        "n_total": len(impr_pcts) if impr_pcts else 0,
        "mean_angle": sum(angle_means) / len(angle_means) if angle_means else 0,
        "ppl_fp16": ppl_fp16, "ppl_rtn": ppl_rtn, "ppl_rotor": ppl_rotor,
        "rotor_delta_ppl": ppl_rotor - ppl_rtn,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose rotor behavior on GPT-2 models")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--n-bits", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = diagnose_model(
        args.model, n_bits=args.n_bits, n_steps=args.n_steps,
        n_eval_batches=args.eval_batches, batch_size=args.batch_size,
        max_length=args.max_length, device=device,
    )

    s = results["summary"]
    print(f"\n{'='*60}")
    print(f"RESULT: {results['model_name']}")
    print(f"{'='*60}")
    print(f"  Params:           {results['n_params']:,}")
    print(f"  Hidden dim:       {results['hidden_dim']}")
    print(f"  Blocks:           {results['n_blocks']}")
    print(f"  Mean improvement: {s['mean_improvement']:+.1f}%")
    print(f"  Layers improved:  {s['n_improved']}/{s['n_total']}")
    print(f"  Mean angle:       {s['mean_angle']:.4f} rad")
    print(f"  PPL FP16:         {s['ppl_fp16']:.2f}")
    print(f"  PPL RTN:          {s['ppl_rtn']:.2f}")
    print(f"  PPL Rotor:        {s['ppl_rotor']:.2f}")
    print(f"  Rotor ΔPPL:       {s['rotor_delta_ppl']:+.2f}")


if __name__ == "__main__":
    main()
