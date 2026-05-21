"""Benchmark GPT-2 post-training quantization with learned rotations.

The core path compares:
  - FP16 baseline
  - RTN (round-to-nearest) weight quantization
  - learned block-diagonal rotor quantization
  - optional QuaRot-style absorption of the learned rotation

Experimental branches such as activation quantization, per-grade quantization,
and ensemble quantization are available only behind ``--experimental``.
"""

import argparse
import math
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .quantization import UniformQuantizer, quantization_error
from .rotor_quant import (
    get_optimal_rotation,
    get_optimal_reflection,
    optimize_rotor_scale,
    apply_rotor_scale,
    finalize_rotor_scale,
)
from .ga import bivector_exp
from .ga import block_diag_rotor_matrix


# ═══════════════════════════════════════════════════════════════════════
#  Activation Quantization via Forward Hooks
# ═══════════════════════════════════════════════════════════════════════

class ActivationQuantizer:
    """Adds forward pre-hooks to quantize activations at each linear/Conv1D layer.

    When weights have been rotated (W' = W @ R^T), the forward pass becomes:
        y = x @ W'^T = x @ R @ W^T
    The activation entering the matmul is implicitly x @ R (rotated).
    Quantizing this rotated activation reduces quantization error because
    the rotation makes the activation distribution more uniform.

    For evaluation with quantized activations:
        y ≈ Q_act(x @ R) @ Q_W(W'^T)

    Usage:
        quantizer = ActivationQuantizer(n_bits=4)
        quantizer.register(model, layers)
        ppl = evaluate_perplexity(model, tokenizer, ...)
        quantizer.remove()
    """

    def __init__(self, n_bits: int = 4, symmetric: bool = True):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        # Activations use per-tensor quantization (dynamic, per-batch)
        self.quantizer = UniformQuantizer(n_bits, symmetric, per_channel=False)
        self.layer_count = 0

    def _make_hook(self, layer_name: str):
        """Create a forward pre-hook that quantizes the input tensor."""
        quantizer_sym = UniformQuantizer(self.n_bits, symmetric=self.symmetric, per_channel=False)

        def hook(module, input):
            x = input[0]
            if x.numel() > 1:
                x_q = quantizer_sym(x)
                return (x_q,) + input[1:]
            return input

        return hook

    def register(self, model: torch.nn.Module,
                 layers: List[Tuple[str, torch.nn.Module, str]]) -> None:
        """Register activation quantization hooks on all specified layers.

        Args:
            model: The model to quantize activations for.
            layers: List of (name, module, layer_type) tuples from get_linear_layers.
        """
        self.remove()  # Clear any existing hooks
        for name, module, ltype in layers:
            handle = module.register_forward_pre_hook(self._make_hook(name))
            self.handles.append(handle)
            self.layer_count += 1

    def register_on_names(self, model: torch.nn.Module,
                          named_modules: List[Tuple[str, torch.nn.Module]]) -> None:
        """Register hooks on modules by (name, module) pairs.

        Args:
            model: The model (unused, kept for API consistency).
            named_modules: List of (name, module) tuples.
        """
        self.remove()
        for name, module in named_modules:
            handle = module.register_forward_pre_hook(self._make_hook(name))
            self.handles.append(handle)
            self.layer_count += 1

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layer_count = 0

    def __repr__(self) -> str:
        return f"ActivationQuantizer({self.n_bits}b, {self.layer_count} layers)"


# ═══════════════════════════════════════════════════════════════════════
#  Layer Utilities
# ═══════════════════════════════════════════════════════════════════════


def _is_conv1d(module: torch.nn.Module) -> bool:
    """Check if a module is a HuggingFace Conv1D (GPT-2 style)."""
    return (
        hasattr(module, "nf") and
        hasattr(module, "weight") and
        module.weight.dim() == 2 and
        hasattr(module, "bias")
    )


def get_linear_layers(
    model: torch.nn.Module,
    prefix: str = "",
    exclude_embeddings: bool = True,
) -> List[Tuple[str, torch.nn.Module, str]]:
    """Recursively find all linear/Conv1D weight layers in a model.

    Args:
        model: The PyTorch model.
        prefix: Name prefix for recursive calls.
        exclude_embeddings: If True, skip layers that are tied to
            token embeddings (lm_head in GPT-2).

    Returns:
        List of (name, module, layer_type) tuples where layer_type is
        'linear' (torch.nn.Linear) or 'conv1d' (HuggingFace Conv1D).
    """
    layers = []
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, torch.nn.Linear):
            if exclude_embeddings and full_name in ("lm_head",):
                continue  # Skip lm_head (tied to token embeddings)
            layers.append((full_name, child, "linear"))
        elif _is_conv1d(child):
            layers.append((full_name, child, "conv1d"))
        else:
            layers.extend(get_linear_layers(child, full_name, exclude_embeddings))
    return layers


def get_weight(module: torch.nn.Module, layer_type: str) -> torch.Tensor:
    """Get the weight matrix in standard (out_dim, in_dim) format."""
    if layer_type == "conv1d":
        # Conv1D stores weight as (in_dim, out_dim) — transpose to standard
        return module.weight.data.T.clone()
    else:
        return module.weight.data.clone()


def set_weight(module: torch.nn.Module, layer_type: str, W: torch.Tensor) -> None:
    """Set the weight matrix from standard (out_dim, in_dim) format."""
    if layer_type == "conv1d":
        # Conv1D expects (in_dim, out_dim) — transpose from standard
        module.weight.data = W.T.to(module.weight.dtype)
    else:
        module.weight.data = W.to(module.weight.dtype)


@torch.no_grad()
def collect_layer_calibration_pairs(
    model: torch.nn.Module,
    tokenizer,
    layers: List[Tuple[str, torch.nn.Module, str]],
    num_batches: int = 4,
    batch_size: int = 2,
    max_length: int = 128,
    device: torch.device = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Collect per-layer input/target activations for calibration.

    Each entry maps a linear/Conv1D layer name to a pair of 2D tensors:
    ``(inputs, targets)`` where rows are flattened token positions.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        print(f"  Could not load WikiText-2 calibration data: {e}")
        return {}

    texts = [t for t in dataset["text"] if len(t.strip()) > 10]
    captures: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {"inputs": [], "targets": []} for name, _, _ in layers
    }
    handles = []

    def make_hook(layer_name: str):
        def hook(module, input, output):
            if not input:
                return
            x = input[0].detach()
            y = output[0].detach() if isinstance(output, (tuple, list)) else output.detach()
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            else:
                x = x.reshape(-1, x.shape[-1])
            if y.dim() > 2:
                y = y.reshape(-1, y.shape[-1])
            else:
                y = y.reshape(-1, y.shape[-1])
            captures[layer_name]["inputs"].append(x.cpu())
            captures[layer_name]["targets"].append(y.cpu())

        return hook

    model.eval()
    for name, module, _ in layers:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        for start_idx in range(0, min(len(texts), num_batches * batch_size), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            if not batch_texts:
                break

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            if encoded["input_ids"].size(1) < 2:
                continue

            model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    calibration_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, data in captures.items():
        if not data["inputs"] or not data["targets"]:
            continue
        x = torch.cat(data["inputs"], dim=0)
        y = torch.cat(data["targets"], dim=0)
        calibration_pairs[name] = (x, y)

    return calibration_pairs


def _layer_optimization_steps(
    layer_name: Optional[str],
    base_steps: int,
    hotspot_steps: int = 0,
    hotspot_regex: str = r"(?:attn|mlp)\.c_proj$",
) -> int:
    """Return a per-layer optimization budget with an optional hotspot boost."""
    if layer_name and hotspot_steps > base_steps and re.search(hotspot_regex, layer_name):
        return hotspot_steps
    return base_steps


def _projection_rank_candidates(target_rank: int, dim: int) -> List[int]:
    """Build a small candidate set around a target projection rank."""
    raw = {
        1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768,
        target_rank,
        target_rank - 64,
        target_rank - 32,
        target_rank - 16,
        target_rank + 16,
        target_rank + 32,
        target_rank + 64,
    }
    return sorted({max(1, min(dim, int(r))) for r in raw if r is not None})


def _low_rank_approximate_matrix(
    M: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Return the best rank-``rank`` approximation of a matrix via SVD."""
    if rank <= 0:
        return torch.zeros_like(M)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, S.numel())
    if k <= 0:
        return torch.zeros_like(M)
    return (U[:, :k] * S[:k]) @ Vh[:k, :]


def _fit_residual_correction_matrix(
    x_train: torch.Tensor,
    target_residual: torch.Tensor,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Fit a residual correction matrix with ridge-regularized least squares.

    Solves X @ A ~= target_residual and returns the implied weight correction
    matrix A^T in standard (out_dim, in_dim) form.
    """
    if x_train is None or target_residual is None:
        return None
    x = x_train.float().reshape(-1, x_train.shape[-1])
    t = target_residual.float().reshape(-1, target_residual.shape[-1])
    if x.numel() == 0 or t.numel() == 0:
        return None

    if ridge > 0:
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, math.sqrt(ridge) * eye], dim=0)
        t_aug = torch.cat(
            [t, torch.zeros(eye.shape[0], t.shape[1], device=t.device, dtype=t.dtype)],
            dim=0,
        )
        sol = torch.linalg.lstsq(x_aug, t_aug).solution
    else:
        sol = torch.linalg.lstsq(x, t).solution
    return sol.T.contiguous()


def _split_calibration_rows(
    inputs: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    train_frac: float = 0.8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Split calibration rows into train/validation portions."""
    if inputs is None or targets is None:
        return None, None, None, None
    n = min(inputs.shape[0], targets.shape[0])
    if n <= 1:
        return inputs, targets, inputs, targets
    n_train = max(1, min(n - 1, int(round(n * train_frac))))
    if n_train >= n:
        n_train = n - 1
    return (
        inputs[:n_train],
        targets[:n_train],
        inputs[n_train:n],
        targets[n_train:n],
    )


@torch.no_grad()
def collect_calibration_batches(
    tokenizer,
    num_batches: int = 4,
    batch_size: int = 2,
    max_length: int = 128,
    device: torch.device = None,
) -> List[Dict[str, torch.Tensor]]:
    """Collect tokenized calibration batches for task-level scoring."""
    from datasets import load_dataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        print(f"  Could not load WikiText-2 calibration batches: {e}")
        return []

    texts = [t for t in dataset["text"] if len(t.strip()) > 10]
    batches: List[Dict[str, torch.Tensor]] = []
    for start_idx in range(0, min(len(texts), num_batches * batch_size), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        if not batch_texts:
            break
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if encoded["input_ids"].size(1) < 2:
            continue
        batches.append({k: v.to(device) for k, v in encoded.items()})
    return batches


@torch.no_grad()
def evaluate_loss_on_batches(
    model: torch.nn.Module,
    batches: List[Dict[str, torch.Tensor]],
) -> float:
    """Compute average causal LM loss on pre-tokenized batches."""
    if not batches:
        return float("inf")
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for encoded in batches:
        input_ids = encoded["input_ids"]
        if input_ids.size(1) < 2:
            continue
        labels = input_ids.clone()
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            pad_mask = attention_mask == 0
            labels[pad_mask] = -100
            n_tokens = attention_mask.sum().item()
        else:
            n_tokens = input_ids.numel()
        outputs = model(
            input_ids.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            labels=labels.to(device),
        )
        loss = outputs.loss
        n_predicted = max(n_tokens - input_ids.shape[0], 1)
        total_loss += loss.item() * n_predicted
        total_tokens += n_predicted
    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


# ═══════════════════════════════════════════════════════════════════════
#  Quantization Functions
# ═══════════════════════════════════════════════════════════════════════


def quantize_weight_rtn(
    W: torch.Tensor,
    n_bits: int = 4,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix with RTN (no rotation).

    Accepts **kwargs for compatibility with replace_linear_weights.

    Returns:
        (quantized_weight, info_dict) matching the rotor function signature.
    """
    q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
    W_q = q(W)
    info = {
        "mode": "rtn",
        "in_dim": W.shape[1],
        "out_dim": W.shape[0],
        "nmse_rtn": quantization_error(W, W_q).item(),
        "time": 0.0,
    }
    return W_q, info


def _choose_rotor_mode(in_dim: int) -> str:
    """Choose the rotor mode based on input dimension.

    Full bivector-exp (torch.matrix_exp) only works with gradients
    for dim <= 64 on this CUDA/PyTorch version. Block-diagonal rotors
    work at any dimension and are gradient-friendly.
    """
    return "block" if in_dim > 64 else "full"


def quantize_weight_with_rotor(
    W: torch.Tensor,
    n_bits: int = 4,
    n_optimization_steps: int = 50,
    n_restarts: int = 2,
    verbose: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix using an optimized block-diagonal rotor.

    Uses block-diagonal 2D rotors by default (works at any dimension,
    gradient-friendly); falls back to full bivector-exp for dim <= 64.

    Returns:
        (quantized_weight, info_dict) where info_dict contains:
          - mode: 'block_rotor', 'full_rotor', or 'rtn_fallback'
          - nmse_rtn, nmse_rotor, improvement_pct
          - in_dim, out_dim, time (seconds)
    """
    device = W.device
    in_dim = W.shape[1]

    t0 = time.time()

    if in_dim <= 2:
        W_q, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": "dim <= 2"}
        return W_q, info

    mode = _choose_rotor_mode(in_dim)

    try:
        R = get_optimal_rotation(
            W, mode=mode, n_bits=n_bits,
            n_optimization_steps=n_optimization_steps,
            verbose=verbose
        )

        q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
        W_rotated = W @ R.T
        W_q = q(W_rotated) @ R

        W_rtn_tensor, _ = quantize_weight_rtn(W, n_bits)
        nmse_rtn = quantization_error(W, W_rtn_tensor).item()
        nmse_rotor = quantization_error(W, W_q).item()
        improvement = (nmse_rtn - nmse_rotor) / nmse_rtn * 100

        mode_label = "block_rotor" if mode == "block" else "full_rotor"
        info = {
            "mode": mode_label,
            "time": time.time() - t0,
            "in_dim": in_dim,
            "out_dim": W.shape[0],
            "nmse_rtn": nmse_rtn,
            "nmse_rotor": nmse_rotor,
            "improvement_pct": improvement,
        }
        return W_q, info

    except Exception as e:
        import traceback
        W_q_tensor, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": f"{type(e).__name__}: {e}"}
        return W_q_tensor, info


def quantize_weight_with_reflection(
    W: torch.Tensor,
    n_bits: int = 4,
    n_optimization_steps: int = 50,
    n_restarts: int = 2,
    verbose: bool = False,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix using a learned Householder reflection."""
    t0 = time.time()
    in_dim = W.shape[1]

    if in_dim <= 1:
        W_q, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": "dim <= 1"}
        return W_q, info

    try:
        H = get_optimal_reflection(
            W,
            n_bits=n_bits,
            n_optimization_steps=n_optimization_steps,
            verbose=verbose,
            calibration_inputs=calibration_inputs,
            calibration_targets=calibration_targets,
            n_restarts=n_restarts,
        )
        q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
        W_reflected = W @ H.T
        W_q = q(W_reflected) @ H

        W_rtn_tensor, _ = quantize_weight_rtn(W, n_bits)
        nmse_rtn = quantization_error(W, W_rtn_tensor).item()
        nmse_reflect = quantization_error(W, W_q).item()
        improvement = (nmse_rtn - nmse_reflect) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0

        info = {
            "mode": "householder_reflection",
            "time": time.time() - t0,
            "in_dim": in_dim,
            "out_dim": W.shape[0],
            "nmse_rtn": nmse_rtn,
            "nmse_rotor": nmse_reflect,
            "improvement_pct": improvement,
            "basis": "calibration_output" if calibration_inputs is not None and calibration_targets is not None else "weight_mse",
        }
        if calibration_inputs is not None and calibration_targets is not None:
            info["calibration_loss"] = F.mse_loss(calibration_inputs @ W_q.T, calibration_targets).item()
        return W_q, info

    except Exception as e:
        W_q_tensor, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": f"{type(e).__name__}: {e}"}
        return W_q_tensor, info


def quantize_weight_with_rotor_scale(
    W: torch.Tensor,
    n_bits: int = 4,
    n_optimization_steps: int = 50,
    n_restarts: int = 2,
    verbose: bool = False,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    learn_scale: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix using learned rotor plus diagonal scaling."""
    t0 = time.time()
    in_dim = W.shape[1]

    if in_dim <= 2:
        W_q, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": "dim <= 2"}
        return W_q, info

    try:
        angles, scale = optimize_rotor_scale(
            W,
            n_bits=n_bits,
            n_optimization_steps=n_optimization_steps,
            lr=0.05,
            verbose=verbose,
            n_restarts=n_restarts,
            calibration_inputs=calibration_inputs,
            calibration_targets=calibration_targets,
            learn_scale=learn_scale,
        )
        R = block_diag_rotor_matrix(angles, in_dim)
        W_rotated = apply_rotor_scale(W, R, scale)
        q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
        W_q = q(W_rotated)
        W_final = finalize_rotor_scale(W_q, R, scale)

        W_rtn_tensor, _ = quantize_weight_rtn(W, n_bits)
        nmse_rtn = quantization_error(W, W_rtn_tensor).item()
        nmse_rotor = quantization_error(W, W_final).item()
        improvement = (nmse_rtn - nmse_rotor) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0

        info = {
            "mode": "block_rotor_scale",
            "time": time.time() - t0,
            "in_dim": in_dim,
            "out_dim": W.shape[0],
            "nmse_rtn": nmse_rtn,
            "nmse_rotor": nmse_rotor,
            "improvement_pct": improvement,
            "scale_mean": scale.mean().item(),
            "scale_min": scale.min().item(),
            "scale_max": scale.max().item(),
        }
        if calibration_inputs is not None and calibration_targets is not None:
            info["calibration_loss"] = F.mse_loss(
                calibration_inputs @ W_final.T, calibration_targets
            ).item()
        return W_final, info

    except Exception as e:
        W_q_tensor, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": f"{type(e).__name__}: {e}"}
        return W_q_tensor, info


def quantize_weight_with_projection_residual(
    W: torch.Tensor,
    n_bits: int = 4,
    residual_bits: int = 8,
    projection_energy: float = 0.9,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    calibration_train_frac: float = 0.8,
    task_eval_fn: Optional[Callable[[torch.Tensor], float]] = None,
    verbose: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix by splitting it into projection and residual parts.

    If calibration inputs are provided, use their covariance to identify the
    dominant input subspace. Otherwise fall back to the right-singular basis
    of the weight matrix.
    """
    t0 = time.time()
    in_dim = W.shape[1]
    try:
        Wf = W.float()
        if Wf.numel() == 0:
            return quantize_weight_rtn(W, n_bits)

        x_fit, y_fit, x_val, y_val = _split_calibration_rows(
            calibration_inputs, calibration_targets, train_frac=calibration_train_frac
        )

        if x_fit is not None and x_fit.numel() > 0:
            X = x_fit.float().reshape(-1, in_dim)
            X = X - X.mean(dim=0, keepdim=True)
            cov = (X.T @ X) / max(X.shape[0], 1)
            evals, evecs = torch.linalg.eigh(cov)
            order = torch.argsort(evals, descending=True)
            evals = evals[order]
            evecs = evecs[:, order]
            energy = evals.clamp(min=0)
        else:
            _, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
            if S.numel() == 0:
                return quantize_weight_rtn(W, n_bits)
            energy = S.pow(2)
            evecs = Vh.T

        total_energy = energy.sum().clamp(min=1e-12)
        cum_energy = torch.cumsum(energy, dim=0) / total_energy
        target_rank = int((cum_energy < projection_energy).sum().item()) + 1
        target_rank = max(1, min(target_rank, evecs.shape[1]))

        candidate_ranks = [target_rank]
        if task_eval_fn is not None:
            candidate_ranks = sorted({
                max(1, target_rank // 2),
                target_rank,
                min(evecs.shape[1], target_rank + 64),
            })
        elif x_fit is not None and y_fit is not None and x_val is not None and y_val is not None:
            candidate_ranks = _projection_rank_candidates(target_rank, evecs.shape[1])

        x_train = None
        y_train = None
        x_val = None
        y_val = None
        if x_fit is not None and y_fit is not None:
            x_train = x_fit.float().reshape(-1, in_dim)
            y_train = y_fit.float().reshape(-1, W.shape[0])
        if calibration_inputs is not None and calibration_targets is not None:
            if x_val is None or y_val is None:
                _, _, x_val_raw, y_val_raw = _split_calibration_rows(
                    calibration_inputs, calibration_targets, train_frac=calibration_train_frac
                )
                x_val = x_val_raw.float().reshape(-1, in_dim) if x_val_raw is not None else None
                y_val = y_val_raw.float().reshape(-1, W.shape[0]) if y_val_raw is not None else None
            else:
                x_val = x_val.float().reshape(-1, in_dim) if x_val is not None else None
                y_val = y_val.float().reshape(-1, W.shape[0]) if y_val is not None else None

        best_loss = float("inf")
        best_k = target_rank
        best_W_q = None
        best_retained_energy = (energy[:target_rank].sum() / total_energy).item()

        for k in candidate_ranks:
            V_k = evecs[:, :k]
            P = V_k @ V_k.T
            W_proj = Wf @ P
            W_resid = Wf - W_proj

            q_proj = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_proj)
            q_resid = UniformQuantizer(residual_bits, symmetric=True, per_channel=True)(W_resid)
            W_q_candidate = (q_proj + q_resid).to(W.dtype)

            if task_eval_fn is not None:
                loss = task_eval_fn(W_q_candidate)
            elif x_val is not None and y_val is not None:
                loss = F.mse_loss(x_val @ W_q_candidate.T, y_val).item()
            else:
                loss = quantization_error(W, W_q_candidate).item()

            if loss < best_loss:
                best_loss = loss
                best_k = k
                best_W_q = W_q_candidate
                best_retained_energy = (energy[:k].sum() / total_energy).item()

        assert best_W_q is not None
        W_q = best_W_q

        W_rtn_tensor, _ = quantize_weight_rtn(W, n_bits)
        nmse_rtn = quantization_error(W, W_rtn_tensor).item()
        nmse_proj = quantization_error(W, W_q).item()
        improvement = (nmse_rtn - nmse_proj) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0
        info = {
            "mode": "projection_residual",
            "time": time.time() - t0,
            "in_dim": in_dim,
            "out_dim": W.shape[0],
            "projection_rank": best_k,
            "retained_energy": best_retained_energy,
            "basis": "activation_covariance" if x_train is not None and x_train.numel() > 0 else "weight_svd",
            "residual_bits": residual_bits,
            "selection_metric": (
                "heldout_next_token_loss" if task_eval_fn is not None else
                "heldout_calibration_loss" if x_val is not None and y_val is not None else
                "quantization_error"
            ),
            "selection_loss": best_loss,
            "nmse_rtn": nmse_rtn,
            "nmse_rotor": nmse_proj,
            "improvement_pct": improvement,
        }
        return W_q, info
    except Exception as e:
        W_q_tensor, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": f"{type(e).__name__}: {e}"}
        return W_q_tensor, info


def quantize_weight_with_projection_residual_model(
    W: torch.Tensor,
    n_bits: int = 4,
    projection_energy: float = 0.9,
    residual_rank: int = 8,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    calibration_train_frac: float = 0.8,
    verbose: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix with an explicit low-rank residual model.

    The layer is decomposed into:
      1. a projection onto a quantization-friendly subspace
      2. uniform quantization of that projected component
      3. a learned low-rank residual correction fit on calibration data

    This is the first live subspace experiment after the archived
    projection-residual split.
    """
    t0 = time.time()
    in_dim = W.shape[1]
    out_dim = W.shape[0]

    try:
        Wf = W.float()
        if Wf.numel() == 0:
            return quantize_weight_rtn(W, n_bits)

        x_fit, y_fit, x_val, y_val = _split_calibration_rows(
            calibration_inputs, calibration_targets, train_frac=calibration_train_frac
        )

        if x_fit is not None and x_fit.numel() > 0:
            X = x_fit.float().reshape(-1, in_dim)
            X = X - X.mean(dim=0, keepdim=True)
            cov = (X.T @ X) / max(X.shape[0], 1)
            evals, evecs = torch.linalg.eigh(cov)
            order = torch.argsort(evals, descending=True)
            evals = evals[order]
            evecs = evecs[:, order]
            basis_kind = "activation_covariance"
            energy = evals.clamp(min=0)
        else:
            _, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
            if S.numel() == 0:
                return quantize_weight_rtn(W, n_bits)
            evecs = Vh.T
            energy = S.pow(2)
            basis_kind = "weight_svd"

        total_energy = energy.sum().clamp(min=1e-12)
        cum_energy = torch.cumsum(energy, dim=0) / total_energy
        target_rank = int((cum_energy < projection_energy).sum().item()) + 1
        target_rank = max(1, min(target_rank, evecs.shape[1]))

        projection_candidates = _projection_rank_candidates(target_rank, evecs.shape[1])
        residual_candidates = sorted({
            1,
            2,
            4,
            8,
            residual_rank,
            max(1, residual_rank // 2),
            min(residual_rank * 2, min(in_dim, out_dim)),
        })
        residual_candidates = [
            max(1, min(min(in_dim, out_dim), int(r))) for r in residual_candidates
        ]
        residual_candidates = sorted(set(residual_candidates))

        if verbose:
            print(
                f"  [proj+resid] basis={basis_kind}, proj_candidates={projection_candidates}, "
                f"resid_candidates={residual_candidates}"
            )

        x_train = None if x_fit is None else x_fit.float().reshape(-1, in_dim)
        y_train = None if y_fit is None else y_fit.float().reshape(-1, out_dim)
        x_val = None if x_val is None else x_val.float().reshape(-1, in_dim)
        y_val = None if y_val is None else y_val.float().reshape(-1, out_dim)

        best_loss = float("inf")
        best_proj_rank = target_rank
        best_resid_rank = residual_rank
        best_W_q = None
        best_retained_energy = (energy[:target_rank].sum() / total_energy).item()

        for k in projection_candidates:
            V_k = evecs[:, :k]
            P = V_k @ V_k.T
            W_proj = Wf @ P
            W_proj_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_proj)

            if x_train is not None and y_train is not None:
                train_residual = y_train - x_train @ W_proj_q.T
                residual_fit = _fit_residual_correction_matrix(x_train, train_residual)
                if residual_fit is None:
                    residual_fit = torch.zeros_like(W_proj_q)
            else:
                residual_fit = Wf - W_proj_q

            for r in residual_candidates:
                residual_low_rank = _low_rank_approximate_matrix(residual_fit, r)
                W_candidate = (W_proj_q + residual_low_rank).to(W.dtype)

                if x_val is not None and y_val is not None:
                    loss = F.mse_loss(x_val @ W_candidate.T, y_val).item()
                else:
                    loss = quantization_error(W, W_candidate).item()

                if loss < best_loss:
                    best_loss = loss
                    best_proj_rank = k
                    best_resid_rank = r
                    best_W_q = W_candidate
                    best_retained_energy = (energy[:k].sum() / total_energy).item()

        assert best_W_q is not None
        W_q = best_W_q

        W_rtn_tensor, _ = quantize_weight_rtn(W, n_bits)
        nmse_rtn = quantization_error(W, W_rtn_tensor).item()
        nmse_model = quantization_error(W, W_q).item()
        improvement = (nmse_rtn - nmse_model) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0
        info = {
            "mode": "projection_residual_model",
            "time": time.time() - t0,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "projection_rank": best_proj_rank,
            "residual_rank": best_resid_rank,
            "retained_energy": best_retained_energy,
            "basis": basis_kind,
            "selection_metric": "heldout_calibration_loss" if x_val is not None and y_val is not None else "quantization_error",
            "selection_loss": best_loss,
            "nmse_rtn": nmse_rtn,
            "nmse_rotor": nmse_model,
            "improvement_pct": improvement,
        }
        return W_q, info
    except Exception as e:
        W_q_tensor, info_rtn = quantize_weight_rtn(W, n_bits)
        info = {**info_rtn, "mode": "rtn_fallback", "time": time.time() - t0,
                "error": f"{type(e).__name__}: {e}"}
        return W_q_tensor, info


def quantize_weight_with_grade_allocation(
    W: torch.Tensor,
    n_bits: int = 4,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    calibration_train_frac: float = 0.8,
    candidate_bit_maps: Optional[List[Dict]] = None,
    bit_cost_weight: float = 1e-3,
    verbose: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix with calibration-guided grade-aware bit allocation.

    This is the constrained successor to the archived per-grade branch:
    instead of hard-coding a single grade split, it searches a small set of
    candidate bit maps and chooses the one with the best held-out calibration
    loss under a mild bit-cost regularizer.
    """
    t0 = time.time()
    try:
        from .experimental.per_grade_quant import quantize_per_grade, _avg_bit_width
    except Exception:
        from .experimental.per_grade_quant import quantize_per_grade, _avg_bit_width

    if candidate_bit_maps is None:
        candidate_bit_maps = [
            {0: 4, 2: 4, "strain": 4},
            {0: 8, 2: 4, "strain": 4},
            {0: 8, 2: 4, "strain": 6},
            {0: 8, 2: 3, "strain": 6},
            {0: 16, 2: 4, "strain": 8},
        ]

    x_fit, y_fit, x_val, y_val = _split_calibration_rows(
        calibration_inputs, calibration_targets, train_frac=calibration_train_frac
    )
    if x_val is None or y_val is None:
        x_val = x_fit
        y_val = y_fit

    best_score = float("inf")
    best_loss = float("inf")
    best_bits = None
    best_map = None
    best_W_q = None
    best_info = None

    for bit_map in candidate_bit_maps:
        W_q, info = quantize_per_grade(W, bit_map, verbose=False)
        if x_val is not None and y_val is not None:
            val_loss = F.mse_loss(x_val @ W_q.T, y_val).item()
        else:
            val_loss = quantization_error(W, W_q).item()
        avg_bits = _avg_bit_width(W.shape[1], bit_map)
        score = val_loss + bit_cost_weight * avg_bits

        if score < best_score:
            best_score = score
            best_loss = val_loss
            best_bits = avg_bits
            best_map = bit_map
            best_W_q = W_q
            best_info = info

    assert best_W_q is not None and best_map is not None and best_info is not None

    W_rtn_tensor, _ = quantize_weight_rtn(W, n_bits)
    nmse_rtn = quantization_error(W, W_rtn_tensor).item()
    nmse_grade = quantization_error(W, best_W_q).item()
    improvement = (nmse_rtn - nmse_grade) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0

    info = {
        "mode": "grade_allocation",
        "time": time.time() - t0,
        "in_dim": W.shape[1],
        "out_dim": W.shape[0],
        "bit_map": {str(k): v for k, v in best_map.items()},
        "avg_bits": best_bits,
        "selection_loss": best_loss,
        "selection_score": best_score,
        "nmse_rtn": nmse_rtn,
        "nmse_rotor": nmse_grade,
        "improvement_pct": improvement,
        "basis": best_info.get("basis", "grade_decomposition"),
        "frac_scalar": best_info.get("frac_scalar", 0.0),
        "frac_bivector": best_info.get("frac_bivector", 0.0),
        "frac_strain": best_info.get("frac_strain", 0.0),
    }
    if x_val is not None and y_val is not None:
        info["calibration_loss"] = best_loss
    return best_W_q, info


# ═══════════════════════════════════════════════════════════════════════
#  QuaRot-style Absorption: Consistent Rotation within Sub-blocks
# ═══════════════════════════════════════════════════════════════════════

def build_absorption_pairs(
    layers: List[Tuple[str, torch.nn.Module, str]]
) -> List[Tuple[Optional[Tuple[str, torch.nn.Module, str]], Tuple[str, torch.nn.Module, str]]]:
    """Build absorption pairs from a linear layer list for GPT-2.

    Identifies pairs where we can apply a consistent rotation:
    - (c_attn, c_proj) within each attention sub-block: SAME rotation
      applied to both so R partially cancels through attention

    Returns:
        List of (prev_layer, curr_layer) tuples.
        prev_layer is None for the first layer in a chain (no absorption).
        curr_layer is the layer whose rotation we optimize.
    """
    # Parse layer names into block index and layer type
    parsed = []
    for name, mod, ltype in layers:
        m = re.match(r"transformer\.h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)", name)
        if m:
            block_idx = int(m.group(1))
            layer_type = m.group(2)
            parsed.append((block_idx, layer_type, name, mod, ltype))
        else:
            # Non-block layers (e.g., lm_head) — no absorption
            parsed.append((-1, name, name, mod, ltype))

    # Sort by block index then by layer type within block
    type_order = {"attn.c_attn": 0, "attn.c_proj": 1, "mlp.c_fc": 2, "mlp.c_proj": 3}

    def sort_key(item):
        block, lt, _, _, _ = item
        return block, type_order.get(lt, 99)

    parsed.sort(key=sort_key)

    pairs = []
    blocks = {}

    # Group by block
    for block_idx, lt, name, mod, ltype in parsed:
        if block_idx < 0:
            pairs.append((None, (name, mod, ltype)))
            continue
        if block_idx not in blocks:
            blocks[block_idx] = {}
        blocks[block_idx][lt] = (name, mod, ltype)

    # Build pairs: (c_attn, c_proj) within each block
    for block_idx in sorted(blocks.keys()):
        blk = blocks[block_idx]
        attn_ca = blk.get("attn.c_attn")
        attn_cp = blk.get("attn.c_proj")
        mlp_fc = blk.get("mlp.c_fc")
        mlp_cp = blk.get("mlp.c_proj")

        # Attention sub-block absorption: learn R from c_attn (curr),
        # apply the same R to c_proj (prev). Since c_attn and c_proj share
        # the same 768-dim input space, R partially cancels through QKV
        # attention computation (QuaRot-style consistent rotation).
        if attn_ca and attn_cp:
            # prev=c_proj, curr=c_attn: we learn R from c_attn (curr)
            pairs.append((attn_cp, attn_ca))

        # MLP sub-block: independent layers (different dims: 768 vs 3072)
        if mlp_fc:
            pairs.append((None, mlp_fc))
        if mlp_cp:
            pairs.append((None, mlp_cp))

    return pairs


def replace_linear_weights_with_absorption(
    model: torch.nn.Module,
    n_bits: int = 4,
    verbose: bool = True,
    n_optimization_steps: int = 50,
    n_restarts: int = 2,
) -> Dict:
    """Quantize GPT-2 with QuaRot-style consistent sub-block rotations.

    For each attention sub-block, learns a rotation R from c_attn and
    applies it to BOTH c_attn and c_proj. This makes R partially cancel
    through the attention path (consistent with QuaRot philosophy).

    When combined with activation quantization (--quantize-activations),
    the activations entering each layer are also quantized. The rotor
    absorption means activations are implicitly rotated (x @ R), making
    them more uniform for quantization.

    MLP layers are quantized independently (different dimensions).

    Args:
        model: The model to quantize in-place.
        n_bits: Bit-width.
        verbose: Print progress.
        n_optimization_steps: Steps per layer.
        n_restarts: Random restarts.

    Returns:
        Dictionary of layer name -> quantization info.
    """
    layers = get_linear_layers(model)
    pairs = build_absorption_pairs(layers)
    all_info = {}

    print(f"  Built {len(pairs)} absorption pairs")

    for idx, (prev_info, curr_info) in enumerate(pairs):
        if curr_info is None:
            continue

        curr_name, curr_mod, curr_type = curr_info
        W_curr = get_weight(curr_mod, curr_type)
        in_dim = W_curr.shape[1]

        if prev_info is not None:
            # Absorption pair: learn R from c_attn (curr), apply same R
            # to c_proj (prev) so rotation partially cancels in attention.
            prev_name, prev_mod, prev_type = prev_info
            W_prev = get_weight(prev_mod, prev_type)

            if verbose:
                state = f"[{idx+1}/{len(pairs)}] {curr_name} (R shared <-> {prev_name})"
                print(f"  {state}")
                print(f"         {curr_name}: ({curr_type}) {W_curr.shape}")
                print(f"         {prev_name}: ({prev_type}) {W_prev.shape} (same R)")

            t0 = time.time()
            try:
                # Step 1: Learn optimal rotation for c_attn (block-diagonal)
                R = get_optimal_rotation(
                    W_curr, mode=_choose_rotor_mode(in_dim), n_bits=n_bits,
                    n_optimization_steps=n_optimization_steps,
                    verbose=False,
                )

                # Step 2: Apply same R to both, quantize both
                q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)

                W_curr_q = q(W_curr @ R.T) @ R  # c_attn rotated + quantized
                W_prev_q = q(W_prev @ R.T) @ R  # c_proj with same R

                set_weight(curr_mod, curr_type, W_curr_q)
                set_weight(prev_mod, prev_type, W_prev_q)

                # NMSE for c_attn (the layer we optimized for)
                W_rtn_tensor, _ = quantize_weight_rtn(W_curr, n_bits)
                nmse_rtn_curr = quantization_error(W_curr, W_rtn_tensor).item()
                nmse_rotor_curr = quantization_error(W_curr, W_curr_q).item()
                impr_curr = (nmse_rtn_curr - nmse_rotor_curr) / nmse_rtn_curr * 100

                # NMSE for c_proj (benefits from the shared R)
                W_rtn_prev, _ = quantize_weight_rtn(W_prev, n_bits)
                nmse_rtn_prev = quantization_error(W_prev, W_rtn_prev).item()
                nmse_rotor_prev = quantization_error(W_prev, W_prev_q).item()
                impr_prev = (nmse_rtn_prev - nmse_rotor_prev) / nmse_rtn_prev * 100

                elapsed = time.time() - t0

                all_info[curr_name] = {
                    "mode": "absorbed_rotor_ca",
                    "time": elapsed,
                    "in_dim": W_curr.shape[1],
                    "out_dim": W_curr.shape[0],
                    "nmse_rtn": nmse_rtn_curr,
                    "nmse_rotor": nmse_rotor_curr,
                    "improvement_pct": impr_curr,
                    "sibling": prev_name,
                }
                all_info[prev_name] = {
                    "mode": "absorbed_rotor_cp",
                    "time": elapsed,
                    "in_dim": W_prev.shape[1],
                    "out_dim": W_prev.shape[0],
                    "nmse_rtn": nmse_rtn_prev,
                    "nmse_rotor": nmse_rotor_prev,
                    "improvement_pct": impr_prev,
                    "sibling": curr_name,
                }

                if verbose:
                    print(f"         [c_attn] NMSE: RTN={nmse_rtn_curr:.6f} -> "
                          f"Rotor={nmse_rotor_curr:.6f} ({impr_curr:+.1f}%)")
                    print(f"         [c_proj] NMSE: RTN={nmse_rtn_prev:.6f} -> "
                          f"Rotor={nmse_rotor_prev:.6f} ({impr_prev:+.1f}%)")
                    print(f"         Done in {elapsed:.1f}s")

            except Exception as e:
                # Fallback: RTN for both
                import traceback
                W_curr_rtn, _ = quantize_weight_rtn(W_curr, n_bits)
                W_prev_rtn, _ = quantize_weight_rtn(W_prev, n_bits)
                set_weight(curr_mod, curr_type, W_curr_rtn)
                set_weight(prev_mod, prev_type, W_prev_rtn)

                all_info[curr_name] = {
                    "mode": "rtn_fallback",
                    "time": time.time() - t0,
                    "error": f"{type(e).__name__}: {e}",
                }
                all_info[prev_name] = {"mode": "rtn_fallback_absorbed_pair"}

                if verbose:
                    print(f"         Fallback to RTN ({type(e).__name__})")

        else:
            # No absorption — quantize independently
            if verbose:
                state = f"[{idx+1}/{len(pairs)}] {curr_name} (independent)"
                print(f"  {state}")
                print(f"         {curr_name}: ({curr_type}) {W_curr.shape}")

            t0 = time.time()
            try:
                # Try block-diagonal rotor (works at any dimension)
                R = get_optimal_rotation(
                    W_curr, mode=_choose_rotor_mode(in_dim), n_bits=n_bits,
                    n_optimization_steps=n_optimization_steps,
                    verbose=False,
                )
                q_curr = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
                W_rotated = W_curr @ R.T
                W_q = q_curr(W_rotated) @ R
                set_weight(curr_mod, curr_type, W_q)

                W_rtn_tensor, _ = quantize_weight_rtn(W_curr, n_bits)
                nmse_rtn = quantization_error(W_curr, W_rtn_tensor).item()
                nmse_rotor = quantization_error(W_curr, W_q).item()
                improvement = (nmse_rtn - nmse_rotor) / nmse_rtn * 100
                elapsed = time.time() - t0

                info = {
                    "mode": "full_rotor", "time": elapsed,
                    "in_dim": in_dim, "out_dim": W_curr.shape[0],
                    "nmse_rtn": nmse_rtn, "nmse_rotor": nmse_rotor,
                    "improvement_pct": improvement,
                }
                all_info[curr_name] = info
                if verbose:
                    print(f"         NMSE: RTN={nmse_rtn:.6f} -> Rotor={nmse_rotor:.6f} "
                          f"({improvement:+.1f}%) in {elapsed:.1f}s")

            except Exception as e:
                W_rtn, _ = quantize_weight_rtn(W_curr, n_bits)
                set_weight(curr_mod, curr_type, W_rtn)
                all_info[curr_name] = {
                    "mode": "rtn_fallback",
                    "time": time.time() - t0,
                    "error": f"{type(e).__name__}: {e}",
                }
                if verbose:
                    print(f"         Fallback to RTN ({type(e).__name__})")

    return all_info


# ═══════════════════════════════════════════════════════════════════════
#  Generic replace_linear_weights (independent per-layer)
# ═══════════════════════════════════════════════════════════════════════

def replace_linear_weights(
    model: torch.nn.Module,
    quantize_fn,
    n_bits: int = 4,
    verbose: bool = True,
    n_optimization_steps: int = 50,
    n_restarts: int = 2,
) -> Dict:
    """Replace all linear/Conv1D layer weights with quantized versions.

    Uses the provided quantize_fn for each layer independently.
    This is the existing (non-absorption) path.

    Args:
        model: The model to quantize in-place.
        quantize_fn: Function(W, n_bits, ...) -> (W_q, info).
        n_bits: Bit-width for quantization.
        verbose: Print progress.
        n_optimization_steps: Steps per layer for rotor optimization.
        n_restarts: Random restarts per layer.

    Returns:
        Dictionary of layer name -> quantization info.
    """
    layers = get_linear_layers(model)
    all_info = {}

    for i, (name, module, layer_type) in enumerate(layers):
        W_orig = get_weight(module, layer_type)

        if verbose:
            print(f"  [{i+1}/{len(layers)}] {name} ({layer_type}): {W_orig.shape}")

        W_q, info = quantize_fn(
            W_orig, n_bits=n_bits,
            layer_name=name,
            module=module,
            layer_type=layer_type,
            n_optimization_steps=n_optimization_steps,
            n_restarts=n_restarts,
            verbose=False
        )

        set_weight(module, layer_type, W_q)
        all_info[name] = info

        if verbose:
            mode = info["mode"]
            t = info["time"]
            if mode in ("full_rotor", "block_rotor", "block_rotor_scale", "projection_residual", "projection_residual_model", "householder_reflection", "grade_allocation"):
                impr = info.get("improvement_pct", 0)
                if mode == "block_rotor_scale":
                    scale_stats = ""
                    if "scale_mean" in info:
                        scale_stats = (
                            f" scale={info['scale_mean']:.2f}"
                            f"[{info['scale_min']:.2f},{info['scale_max']:.2f}]"
                        )
                    print(f"         NMSE: RTN={info['nmse_rtn']:.6f} -> Rotor+Scale={info['nmse_rotor']:.6f} "
                          f"({impr:+.1f}%) [scale] in {t:.1f}s{scale_stats}")
                elif mode == "projection_residual":
                    rank = info.get("projection_rank", "?")
                    energy = info.get("retained_energy", 0.0)
                    rb = info.get("residual_bits", "?")
                    print(f"         NMSE: RTN={info['nmse_rtn']:.6f} -> Proj+Resid={info['nmse_rotor']:.6f} "
                          f"({impr:+.1f}%) [k={rank}, e={energy:.1%}, r{rb}] in {t:.1f}s")
                elif mode == "projection_residual_model":
                    pr = info.get("projection_rank", "?")
                    rr = info.get("residual_rank", "?")
                    energy = info.get("retained_energy", 0.0)
                    print(f"         NMSE: RTN={info['nmse_rtn']:.6f} -> Proj+ResidualModel={info['nmse_rotor']:.6f} "
                          f"({impr:+.1f}%) [kp={pr}, kr={rr}, e={energy:.1%}] in {t:.1f}s")
                elif mode == "householder_reflection":
                    basis = info.get("basis", "?")
                    print(f"         NMSE: RTN={info['nmse_rtn']:.6f} -> Reflection={info['nmse_rotor']:.6f} "
                          f"({impr:+.1f}%) [{basis}] in {t:.1f}s")
                elif mode == "grade_allocation":
                    bits = info.get("avg_bits", 0.0)
                    bmap = info.get("bit_map", {})
                    print(f"         NMSE: RTN={info['nmse_rtn']:.6f} -> GradeAlloc={info['nmse_rotor']:.6f} "
                          f"({impr:+.1f}%) [~{bits:.1f}b, map={bmap}] in {t:.1f}s")
                else:
                    m = mode[:5]  # 'full_' or 'block'
                    print(f"         NMSE: RTN={info['nmse_rtn']:.6f} -> Rotor={info['nmse_rotor']:.6f} "
                          f"({impr:+.1f}%) [{m}] in {t:.1f}s")
            elif mode.startswith("per_grade"):
                impr = info.get("improvement_pct", 0)
                b0 = info.get("bit_map", {}).get("0", "?")
                b2 = info.get("bit_map", {}).get("2", "?")
                bs = info.get("bit_map", {}).get("strain", "?")
                ns = info.get("nmse_scalar", 0)
                nb = info.get("nmse_bivector", 0)
                nst = info.get("nmse_strain", 0)
                fs = info.get("frac_scalar", 0)
                fb = info.get("frac_bivector", 0)
                fst = info.get("frac_strain", 0)
                print(f"         PG({b0}/{b2}/{bs}) NMSE: S={ns:.2e}({fs:.0%}) B={nb:.2e}({fb:.0%}) St={nst:.2e}({fst:.0%})")
                print(f"         Combined={info.get('nmse_rotor', 0):.6f} ({impr:+.1f}% vs RTN) in {t:.1f}s")
            elif mode == "rtn_fallback":
                err_str = info.get("error", "")
                print(f"         RTN (fallback): {err_str} in {t:.1f}s")
            else:
                print(f"         Done in {t:.1f}s")

    return all_info


# ═══════════════════════════════════════════════════════════════════════
#  Model Loading & Evaluation
# ═══════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str = "gpt2", device: torch.device = None):
    """Load GPT-2 model and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    num_batches: int = 20,
    batch_size: int = 4,
    max_length: int = 128,
    device: torch.device = None,
) -> float:
    """Evaluate perplexity on WikiText-2 validation set.

    Uses proper causal LM evaluation with padding masking.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        return float("nan")

    texts = [t for t in dataset["text"] if len(t.strip()) > 10]

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    n_samples = min(len(texts), num_batches * batch_size)

    for start_idx in range(0, n_samples, batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        if not batch_texts:
            break

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = encoded["input_ids"]

        if input_ids.size(1) < 2:
            continue

        labels = input_ids.clone()
        pad_mask = input_ids == tokenizer.pad_token_id
        labels[pad_mask] = -100

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        n_tokens = (~pad_mask).sum().item()
        n_predicted = max(n_tokens - batch_texts.__len__(), 1)

        total_loss += loss.item() * n_predicted
        total_tokens += n_predicted
        n_batches += 1

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Create a deep copy of a model."""
    import copy
    return copy.deepcopy(model)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Quantize GPT-2 with rotor-based PTQ")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--n-bits", type=int, default=4,
                        help="Quantization bit-width")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Optimization steps per layer")
    parser.add_argument("--n-restarts", type=int, default=1,
                        help="Random restarts (1 = no restarts)")
    parser.add_argument("--eval-batches", type=int, default=50,
                        help="Number of evaluation batches")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max sequence length for evaluation")
    parser.add_argument("--skip-rotor", action="store_true",
                        help="Skip rotor optimization (RTN only)")
    parser.add_argument("--skip-rtn", action="store_true",
                        help="Skip RTN baseline")
    parser.add_argument("--absorb", action="store_true",
                        help="Use QuaRot-style absorption: apply same rotation "
                             "to c_attn and c_proj in each attention sub-block")
    parser.add_argument("--quantize-lm-head", action="store_true",
                        help="Also quantize lm_head (usually skipped)")
    parser.add_argument("--experimental", action="store_true",
                        help="Enable experimental branches such as activation "
                             "quantization, per-grade quantization, and ensemble PTQ")
    parser.add_argument("--rotor-scale", action="store_true",
                        help="Experimental: learn diagonal scaling in addition to the rotor")
    parser.add_argument("--projection-residual", action="store_true",
                        help="Experimental: split layers into projection and residual parts before quantizing")
    parser.add_argument("--projection-residual-model", action="store_true",
                        help="Experimental: projection + explicit low-rank residual model")
    parser.add_argument("--projection-energy", type=float, default=0.9,
                        help="Energy fraction to retain in the projection subspace")
    parser.add_argument("--residual-bits", type=int, default=8,
                        help="Bit-width for the residual branch in projection-residual mode")
    parser.add_argument("--residual-rank", type=int, default=8,
                        help="Rank of the explicit residual model in projection-residual-model mode")
    parser.add_argument("--reflection", action="store_true",
                        help="Experimental: learn a Householder reflection before quantization")
    parser.add_argument("--grade-alloc", action="store_true",
                        help="Experimental: calibration-guided grade-aware bit allocation")
    parser.add_argument("--grade-alloc-regex", type=str, default=r"mlp\.c_proj$",
                        help="Regex for layers eligible for grade-aware allocation")
    parser.add_argument("--grade-bit-cost", type=float, default=1e-3,
                        help="Penalty weight for larger average bit maps in grade allocation")
    parser.add_argument("--hotspot-steps", type=int, default=0,
                        help="Optional extra optimization steps for hotspot projection layers")
    parser.add_argument("--hotspot-regex", type=str, default=r"(?:attn|mlp)\.c_proj$",
                        help="Regex for layers that should receive extra rotor optimization steps")
    parser.add_argument("--per-grade", action="store_true",
                        help="Experimental: use per-grade quantization")
    parser.add_argument("--bit-map", type=str, default=None,
                        help="Bit map for per-grade quantization: e.g. '0:8,2:3,strain:6'")
    parser.add_argument("--quantize-activations", action="store_true",
                        help="Experimental: also quantize activations using forward hooks")
    parser.add_argument("--ensemble", action="store_true",
                        help="Experimental: rotation-aware ensemble quantization")
    parser.add_argument("--submodel-size", type=int, default=2,
                        help="Number of blocks per sub-model for ensemble quantization")
    parser.add_argument("--calibration-batches", type=int, default=4,
                        help="Number of calibration batches for rotor-scale and ensemble calibration")
    parser.add_argument("--calibration-batch-size", type=int, default=2,
                        help="Batch size for calibration data generation")
    args = parser.parse_args()

    experimental_requested = (
        args.per_grade or args.quantize_activations or args.ensemble or
        args.rotor_scale or args.projection_residual or args.projection_residual_model or
        args.reflection or args.grade_alloc or args.hotspot_steps > 0
    )
    if experimental_requested and not args.experimental:
        parser.error("--per-grade, --quantize-activations, --ensemble, --rotor-scale, --projection-residual, --projection-residual-model, --reflection, --grade-alloc, and --hotspot-steps require --experimental")

    print("=" * 70)
    if args.experimental and args.per_grade:
        mode_str = "Per-Grade Quantization"
    elif args.absorb:
        mode_str = "QuaRot-style Absorption"
    elif args.experimental and args.rotor_scale:
        mode_str = "Rotor + Scaling"
    elif args.experimental and args.reflection:
        mode_str = "Reflection"
    elif args.experimental and args.grade_alloc:
        mode_str = "Grade Allocation"
    elif args.experimental and args.projection_residual:
        mode_str = "Projection + Residual"
    elif args.experimental and args.projection_residual_model:
        mode_str = "Projection + Residual Model"
    else:
        mode_str = "Independent Rotor"
    if args.experimental and args.quantize_activations:
        mode_str += " + Activation Quantization"
    print(f"GPT-2 Rotor-Based Quantization  [{mode_str}]")
    print(f"Model: {args.model}, {args.n_bits}-bit, {args.n_steps} steps/layer")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model, tokenizer, device = load_model_and_tokenizer(args.model, device)

    n_params = sum(p.numel() for p in model.parameters())
    exclude_emb = not args.quantize_lm_head
    layers = get_linear_layers(model, exclude_embeddings=exclude_emb)
    print(f"Parameters: {n_params:,}")
    print(f"Linear/Conv1D layers: {len(layers)} "
          f"({'skipping lm_head' if exclude_emb else 'including lm_head'})")
    for name, mod, ltype in layers[:6]:
        W = get_weight(mod, ltype)
        print(f"  {name} ({ltype}): {W.shape}")
    if len(layers) > 6:
        print(f"  ... and {len(layers) - 6} more")

    results = {}

    # ── Baseline: FP16 ────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("1. FP16 Baseline (no quantization)")
    print(f"{'─'*50}")
    model_fp16 = clone_model(model).half().to(device)
    t0 = time.time()
    ppl_fp16 = evaluate_perplexity(
        model_fp16, tokenizer, num_batches=args.eval_batches,
        batch_size=args.batch_size, max_length=args.max_length, device=device
    )
    t_fp16 = time.time() - t0
    results["FP16"] = ppl_fp16
    print(f"  Perplexity: {ppl_fp16:.2f} (in {t_fp16:.1f}s)")
    del model_fp16

    # ── RTN Baseline ──────────────────────────────────────────────────
    if not args.skip_rtn:
        print(f"\n{'─'*50}")
        rtn_label = f"RTN ({args.n_bits}-bit)"
        if args.experimental and args.quantize_activations:
            rtn_label += f" + Act Quant"
        print(f"2. {rtn_label}")
        print(f"{'─'*50}")
        model_rtn = clone_model(model).to(device)
        t0 = time.time()
        rtn_info = replace_linear_weights(
            model_rtn, quantize_weight_rtn, n_bits=args.n_bits, verbose=True
        )

        # Activation quantization hooks (if enabled)
        act_quantizer = None
        if args.experimental and args.quantize_activations:
            act_quantizer = ActivationQuantizer(n_bits=args.n_bits)
            act_quantizer.register(model_rtn, get_linear_layers(model_rtn, exclude_embeddings=exclude_emb))
            print(f"  Registered {act_quantizer.layer_count} activation quantization hooks")

        t_rtn = time.time() - t0
        print(f"\n  Evaluating perplexity...")
        ppl_rtn = evaluate_perplexity(
            model_rtn, tokenizer, num_batches=args.eval_batches,
            batch_size=args.batch_size, max_length=args.max_length, device=device
        )
        if act_quantizer:
            act_quantizer.remove()
        results[rtn_label] = ppl_rtn
        print(f"  Perplexity: {ppl_rtn:.2f} (quantized in {t_rtn:.1f}s)")
        del model_rtn

    # ── Optimized Rotor / Per-Grade Quantization ──────────────────────
    if not args.skip_rotor and not args.per_grade and not args.ensemble:
        print(f"\n{'─'*50}")
        label = "Absorbed Rotor" if args.absorb else "Independent Rotor"
        if args.experimental and args.rotor_scale and not args.absorb:
            label = "Rotor + Scaling"
        elif args.experimental and args.reflection:
            label = "Reflection"
        elif args.experimental and args.grade_alloc:
            label = "Grade Allocation"
        elif args.experimental and args.projection_residual_model:
            label = "Projection + Residual Model"
        elif args.experimental and args.projection_residual:
            label = "Projection + Residual"
        if args.experimental and args.quantize_activations:
            label += " + Act Quant"
        print(f"3. {label} Quantization ({args.n_bits}-bit)")
        print(f"{'─'*50}")

        n_linear_layers = len(layers)
        print(f"  Layers: {n_linear_layers}, Steps: {args.n_steps}, "
              f"Restarts: {args.n_restarts}")

        hotspot_steps = args.hotspot_steps if args.hotspot_steps > 0 else args.n_steps

        def _layer_steps(layer_name: Optional[str]) -> int:
            return _layer_optimization_steps(
                layer_name,
                base_steps=args.n_steps,
                hotspot_steps=hotspot_steps,
                hotspot_regex=args.hotspot_regex,
            )

        calibration_pairs = {}
        task_calibration_batches = []
        if args.experimental and (args.rotor_scale or args.projection_residual or args.projection_residual_model or args.reflection or args.grade_alloc):
            print(
                f"  Collecting calibration pairs "
                f"({args.calibration_batches} batches, batch_size={args.calibration_batch_size})..."
            )
            calib_model = clone_model(model).to(device)
            calib_layers = get_linear_layers(calib_model, exclude_embeddings=exclude_emb)
            calibration_pairs = collect_layer_calibration_pairs(
                calib_model,
                tokenizer,
                calib_layers,
                num_batches=args.calibration_batches,
                batch_size=args.calibration_batch_size,
                max_length=args.max_length,
                device=device,
            )
            del calib_model
            print(f"  Collected calibration pairs for {len(calibration_pairs)}/{len(calib_layers)} layers")

            if args.projection_residual:
                task_calibration_batches = collect_calibration_batches(
                    tokenizer,
                    num_batches=max(2, args.calibration_batches),
                    batch_size=args.calibration_batch_size,
                    max_length=args.max_length,
                    device=device,
                )
                print(f"  Collected {len(task_calibration_batches)} task calibration batches")

        model_rotor = clone_model(model).to(device)
        t0 = time.time()

        if args.absorb:
            rotor_info = replace_linear_weights_with_absorption(
                model_rotor,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )
        elif args.experimental and args.rotor_scale:
            def _rotor_scale_quantize_fn(W, n_bits=4, layer_name=None, **kwargs):
                cal_in, cal_tgt = calibration_pairs.get(layer_name, (None, None))
                if cal_in is not None:
                    cal_in = cal_in.to(W.device, dtype=W.dtype)
                if cal_tgt is not None:
                    cal_tgt = cal_tgt.to(W.device, dtype=W.dtype)
                return quantize_weight_with_rotor_scale(
                    W,
                    n_bits=n_bits,
                    n_optimization_steps=_layer_steps(layer_name),
                    n_restarts=args.n_restarts,
                    verbose=False,
                    calibration_inputs=cal_in,
                    calibration_targets=cal_tgt,
                )

            rotor_info = replace_linear_weights(
                model_rotor,
                _rotor_scale_quantize_fn,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )
        elif args.experimental and args.grade_alloc:
            def _grade_alloc_quantize_fn(W, n_bits=4, layer_name=None, **kwargs):
                cal_in, cal_tgt = calibration_pairs.get(layer_name, (None, None))
                if cal_in is not None:
                    cal_in = cal_in.to(W.device, dtype=W.dtype)
                if cal_tgt is not None:
                    cal_tgt = cal_tgt.to(W.device, dtype=W.dtype)
                if layer_name is not None and not re.search(args.grade_alloc_regex, layer_name):
                    return quantize_weight_rtn(W, n_bits=n_bits)
                return quantize_weight_with_grade_allocation(
                    W,
                    n_bits=n_bits,
                    calibration_inputs=cal_in,
                    calibration_targets=cal_tgt,
                    bit_cost_weight=args.grade_bit_cost,
                    verbose=False,
                )

            rotor_info = replace_linear_weights(
                model_rotor,
                _grade_alloc_quantize_fn,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )
        elif args.experimental and args.projection_residual:
            def _projection_residual_quantize_fn(W, n_bits=4, layer_name=None, **kwargs):
                module = kwargs.get("module")
                layer_type = kwargs.get("layer_type")
                cal_in, cal_tgt = calibration_pairs.get(layer_name, (None, None))
                if cal_in is not None:
                    cal_in = cal_in.to(W.device, dtype=W.dtype)
                if cal_tgt is not None:
                    cal_tgt = cal_tgt.to(W.device, dtype=W.dtype)
                if layer_name is not None and not re.search(r"mlp\.c_proj$", layer_name):
                    return quantize_weight_rtn(W, n_bits=n_bits)
                task_eval_fn = None
                if module is not None and task_calibration_batches:
                    heldout_batches = task_calibration_batches[max(1, len(task_calibration_batches) // 2):]
                    if not heldout_batches:
                        heldout_batches = task_calibration_batches

                    def _task_eval_fn(W_candidate: torch.Tensor) -> float:
                        with torch.no_grad():
                            set_weight(module, layer_type, W_candidate)
                            try:
                                return evaluate_loss_on_batches(model_rotor, heldout_batches)
                            finally:
                                set_weight(module, layer_type, W)

                    task_eval_fn = _task_eval_fn
                return quantize_weight_with_projection_residual(
                    W,
                    n_bits=n_bits,
                    residual_bits=args.residual_bits,
                    projection_energy=args.projection_energy,
                    calibration_inputs=cal_in,
                    calibration_targets=cal_tgt,
                    task_eval_fn=task_eval_fn,
                    verbose=False,
                )

            rotor_info = replace_linear_weights(
                model_rotor,
                _projection_residual_quantize_fn,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )
        elif args.experimental and args.projection_residual_model:
            def _projection_residual_model_quantize_fn(W, n_bits=4, layer_name=None, **kwargs):
                cal_in, cal_tgt = calibration_pairs.get(layer_name, (None, None))
                if cal_in is not None:
                    cal_in = cal_in.to(W.device, dtype=W.dtype)
                if cal_tgt is not None:
                    cal_tgt = cal_tgt.to(W.device, dtype=W.dtype)
                if layer_name is not None and not re.search(r"(?:attn|mlp)\.c_proj$", layer_name):
                    return quantize_weight_rtn(W, n_bits=n_bits)
                return quantize_weight_with_projection_residual_model(
                    W,
                    n_bits=n_bits,
                    projection_energy=args.projection_energy,
                    residual_rank=args.residual_rank,
                    calibration_inputs=cal_in,
                    calibration_targets=cal_tgt,
                    verbose=False,
                )

            rotor_info = replace_linear_weights(
                model_rotor,
                _projection_residual_model_quantize_fn,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )
        elif args.experimental and args.reflection:
            def _reflection_quantize_fn(W, n_bits=4, layer_name=None, **kwargs):
                cal_in, cal_tgt = calibration_pairs.get(layer_name, (None, None))
                if cal_in is not None:
                    cal_in = cal_in.to(W.device, dtype=W.dtype)
                if cal_tgt is not None:
                    cal_tgt = cal_tgt.to(W.device, dtype=W.dtype)
                return quantize_weight_with_reflection(
                    W,
                    n_bits=n_bits,
                    n_optimization_steps=_layer_steps(layer_name),
                    n_restarts=args.n_restarts,
                    verbose=False,
                    calibration_inputs=cal_in,
                    calibration_targets=cal_tgt,
                )

            rotor_info = replace_linear_weights(
                model_rotor,
                _reflection_quantize_fn,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )
        else:
            def _rotor_quantize_fn(W, n_bits=4, layer_name=None, **kwargs):
                return quantize_weight_with_rotor(
                    W,
                    n_bits=n_bits,
                    n_optimization_steps=_layer_steps(layer_name),
                    n_restarts=args.n_restarts,
                    verbose=False,
                )

            rotor_info = replace_linear_weights(
                model_rotor,
                _rotor_quantize_fn,
                n_bits=args.n_bits,
                verbose=True,
                n_optimization_steps=args.n_steps,
                n_restarts=args.n_restarts,
            )

        # Activation quantization hooks (if enabled)
        act_quantizer = None
        if args.experimental and args.quantize_activations:
            act_quantizer = ActivationQuantizer(n_bits=args.n_bits)
            act_quantizer.register(model_rotor, get_linear_layers(model_rotor, exclude_embeddings=exclude_emb))
            print(f"  Registered {act_quantizer.layer_count} activation quantization hooks")

        t_rotor = time.time() - t0
        print(f"\n  Evaluating perplexity...")
        ppl_rotor = evaluate_perplexity(
            model_rotor, tokenizer, num_batches=args.eval_batches,
            batch_size=args.batch_size, max_length=args.max_length, device=device
        )
        if act_quantizer:
            act_quantizer.remove()
        label_k = f"{label} ({args.n_bits}b)"
        results[label_k] = ppl_rotor
        print(f"  Perplexity: {ppl_rotor:.2f} (optimized in {t_rotor:.1f}s)")

        # Summary stats
        improvements = [
            v.get("improvement_pct", 0) for v in rotor_info.values()
            if v.get("improvement_pct") is not None
        ]
        fallbacks = sum(
            1 for v in rotor_info.values()
            if v.get("mode") in ("rtn_fallback",)
        )
        if improvements:
            metric_label = "Rotor" if label.startswith("Rotor") or label.startswith("Independent") or label.startswith("Absorbed") else label
            print(f"\n  {metric_label} NMSE improvement over RTN (lower is better):")
            print(f"    Mean improvement: {sum(improvements)/len(improvements):.1f}%")
            print(f"    Max improvement:  {max(improvements):.1f}%")
            print(f"    Layers improved:  {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
            if fallbacks:
                print(f"    RTN fallbacks:    {fallbacks}/{len(rotor_info)}")
        del model_rotor

    # ── Per-Grade Quantization ────────────────────────────────────────
    if args.experimental and not args.skip_rotor and args.per_grade:
        print(f"\n{'─'*50}")
        bit_map_str = args.bit_map if args.bit_map else "0:8,2:3,strain:6"
        # Parse bit-map string
        bit_map = {}
        for part in bit_map_str.split(","):
            k, v = part.split(":")
            k = k.strip()
            if k == "strain":
                bit_map["strain"] = int(v)
            else:
                bit_map[int(k)] = int(v)
        from .experimental.per_grade_quant import quantize_per_grade, _avg_bit_width
        hidden_dim = model.transformer.h[0].attn.embed_dim
        avg_bits_est = _avg_bit_width(hidden_dim, bit_map)
        label_pg = f"Per-Grade (bit map: {bit_map}, ~{avg_bits_est:.1f}b avg)"
        if args.experimental and args.quantize_activations:
            label_pg += " + Act Quant"
        print(f"3. {label_pg}")
        print(f"{'─'*50}")

        model_pg = clone_model(model).to(device)
        t0 = time.time()

        # Use replace_linear_weights with a custom closure capturing bit_map
        def _per_grade_quantize_fn(W, n_bits=4, **kwargs):
            t_start = time.time()
            W_q, info = quantize_per_grade(W, bit_map, verbose=False)
            info["time"] = time.time() - t_start
            return W_q, info

        pg_info = replace_linear_weights(
            model_pg, _per_grade_quantize_fn, n_bits=args.n_bits,
            verbose=True,
        )

        # Activation quantization hooks (if enabled)
        act_quantizer = None
        if args.experimental and args.quantize_activations:
            act_quantizer = ActivationQuantizer(n_bits=args.n_bits)
            act_quantizer.register(model_pg, get_linear_layers(model_pg, exclude_embeddings=exclude_emb))
            print(f"  Registered {act_quantizer.layer_count} activation quantization hooks")

        t_pg = time.time() - t0
        print(f"\n  Evaluating perplexity...")
        ppl_pg = evaluate_perplexity(
            model_pg, tokenizer, num_batches=args.eval_batches,
            batch_size=args.batch_size, max_length=args.max_length, device=device
        )
        if act_quantizer:
            act_quantizer.remove()
        bit_map_label = f"PG({bit_map.get(0,8)},{bit_map.get(2,3)},{bit_map.get('strain',6)})"
        label_k = f"{bit_map_label}"
        if args.experimental and args.quantize_activations:
            label_k += " + Act"
        results[label_k] = ppl_pg
        print(f"  Perplexity: {ppl_pg:.2f} (in {t_pg:.1f}s)")

        # Summary stats
        improvements = [
            v.get("improvement_pct", 0) for v in pg_info.values()
            if v.get("improvement_pct") is not None
        ]
        if improvements:
            print(f"\n  Per-Grade NMSE improvement over uniform 4b RTN:")
            print(f"    Mean improvement: {sum(improvements)/len(improvements):.1f}%")
            print(f"    Max improvement:  {max(improvements):.1f}%")
            print(f"    Layers improved:  {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

        # Average grade fractions
        frac_scalars = [v.get("frac_scalar", 0) for v in pg_info.values() if "frac_scalar" in v]
        frac_bivectors = [v.get("frac_bivector", 0) for v in pg_info.values() if "frac_bivector" in v]
        frac_strains = [v.get("frac_strain", 0) for v in pg_info.values() if "frac_strain" in v]
        if frac_scalars:
            print(f"\n  Grade power distribution (avg across layers):")
            print(f"    Scalar:   {sum(frac_scalars)/len(frac_scalars):.1%}")
            print(f"    Bivector: {sum(frac_bivectors)/len(frac_bivectors):.1%}")
            print(f"    Strain:   {sum(frac_strains)/len(frac_strains):.1%}")

        # Storage cost note
        b0 = bit_map.get(0, 8)
        b2 = bit_map.get(2, 3)
        bs = bit_map.get('strain', 6)
        storage_bits = b0 + b2 + bs
        print(f"\n  NOTE: Per-grade stores each grade as a full matrix before summing.")
        print(f"  Storage cost: {b0}+{b2}+{bs} = {storage_bits} bits/param vs 4 bits/param (uniform RTN).")
        print(f"  The effective avg ({avg_bits_est:.1f}b) counts only independent parameters.")
        print(f"  A practical implementation could store grade components in packed formats.")
        del model_pg

    # ── Ensemble Quantization (Rotation-Aware Sub-models) ─────────────-
    if args.experimental and args.ensemble:
        from .experimental.ensemble_quant import optimize_ensemble_rotors

        print(f"\n{'─'*50}")
        label_ens = f"Ensemble (submodel={args.submodel_size})"
        if args.experimental and args.quantize_activations:
            label_ens += " + Act Quant"
        print(f"3. {label_ens} Quantization ({args.n_bits}-bit)")
        print(f"{'─'*50}")

        n_blocks = len(model.transformer.h)
        n_submodels = (n_blocks + args.submodel_size - 1) // args.submodel_size
        print(f"  Blocks: {n_blocks}, Sub-models: {n_submodels}, "
              f"Size: {args.submodel_size}, Steps: {args.n_steps}")

        model_ens = clone_model(model).to(device)
        t0 = time.time()

        model_ens, ens_info = optimize_ensemble_rotors(
            model_ens, tokenizer,
            n_bits=args.n_bits,
            n_steps_per_submodel=args.n_steps,
            submodel_size=args.submodel_size,
            n_calibration_batches=args.calibration_batches,
            calibration_batch_size=args.calibration_batch_size,
            max_length=args.max_length,
            verbose=True,
        )

        # Activation quantization hooks (if enabled)
        act_quantizer = None
        if args.experimental and args.quantize_activations:
            act_quantizer = ActivationQuantizer(n_bits=args.n_bits)
            act_quantizer.register(model_ens, get_linear_layers(model_ens, exclude_embeddings=exclude_emb))
            print(f"  Registered {act_quantizer.layer_count} activation quantization hooks")

        t_ens = time.time() - t0
        print(f"\n  Evaluating perplexity...")
        ppl_ens = evaluate_perplexity(
            model_ens, tokenizer, num_batches=args.eval_batches,
            batch_size=args.batch_size, max_length=args.max_length, device=device
        )
        if act_quantizer:
            act_quantizer.remove()
        label_k = f"Ensemble (submodel={args.submodel_size}) ({args.n_bits}b)"
        if args.experimental and args.quantize_activations:
            label_k += " + Act"
        results[label_k] = ppl_ens
        print(f"  Perplexity: {ppl_ens:.2f} (optimized in {t_ens:.1f}s)")

        # Summary stats
        improvements = [
            v.get("improvement_pct", 0) for v in ens_info.values()
            if v.get("improvement_pct") is not None
        ]
        if improvements:
            print(f"\n  Ensemble improvement over RTN (per sub-model):")
            print(f"    Mean improvement: {sum(improvements)/len(improvements):.1f}%")
            print(f"    Sub-models improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
        del model_ens

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for method, ppl in results.items():
        ppl_str = f"{ppl:.2f}" if math.isfinite(ppl) else "N/A"
        print(f"  {method:40s}: PPL = {ppl_str}")
    if "FP16" in results and math.isfinite(results["FP16"]):
        print(f"  {'─'*45}")
        for method, ppl in results.items():
            if method != "FP16" and math.isfinite(ppl):
                degradation = ppl - results["FP16"]
                pct = (ppl / results["FP16"] - 1) * 100
                print(f"  Degradation vs FP16 ({method:27s}): +{degradation:.2f} PPL ({pct:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
