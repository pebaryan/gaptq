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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .quantization import UniformQuantizer, quantization_error
from .rotor_quant import get_optimal_rotation
from .ga import bivector_exp


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
            n_optimization_steps=n_optimization_steps,
            n_restarts=n_restarts,
            verbose=False
        )

        set_weight(module, layer_type, W_q)
        all_info[name] = info

        if verbose:
            mode = info["mode"]
            t = info["time"]
            if mode in ("full_rotor", "block_rotor"):
                impr = info.get("improvement_pct", 0)
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
                        help="Number of calibration batches for ensemble quantization")
    parser.add_argument("--calibration-batch-size", type=int, default=2,
                        help="Batch size for calibration data generation")
    args = parser.parse_args()

    experimental_requested = args.per_grade or args.quantize_activations or args.ensemble
    if experimental_requested and not args.experimental:
        parser.error("--per-grade, --quantize-activations, and --ensemble require --experimental")

    print("=" * 70)
    if args.experimental and args.per_grade:
        mode_str = "Per-Grade Quantization"
    elif args.absorb:
        mode_str = "QuaRot-style Absorption"
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
        if args.experimental and args.quantize_activations:
            label += " + Act Quant"
        print(f"3. {label} Quantization ({args.n_bits}-bit)")
        print(f"{'─'*50}")

        n_linear_layers = len(layers)
        print(f"  Layers: {n_linear_layers}, Steps: {args.n_steps}, "
              f"Restarts: {args.n_restarts}")

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
        else:
            rotor_info = replace_linear_weights(
                model_rotor,
                quantize_weight_with_rotor,
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
            print(f"\n  Rotor NMSE improvement over RTN (lower is better):")
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
