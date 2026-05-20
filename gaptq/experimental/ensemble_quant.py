"""Rotation-aware ensemble quantization.

Instead of optimizing each layer's rotor independently (which disrupts cross-layer
dynamics on larger models), groups consecutive GPT-2 blocks into sub-models and
optimizes ALL rotors within each sub-model JOINTLY to minimize the output MSE
of the sub-model on real calibration data.

This captures cross-layer interactions:
  - Rotors for earlier layers learn to produce activations that later layers'
    rotors can quantize well
  - Degeneracies between rotors in the same sub-model are resolved
  - The optimization objective is the actual task (output reconstruction),
    not a proxy (per-layer weight NMSE)

Key hyperparameter: sub-model size (number of blocks per group).
  - size=1: independent per-layer (degenerates to independent rotor)
  - size=2: 2 blocks, 8 rotors optimized jointly
  - size=4: 4 blocks, 16 rotors optimized jointly
  - size=N: all blocks in one go (full model, very expensive)
"""

import math
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantization import UniformQuantizer
from ..ga import block_diag_rotor_matrix
from ..quantize_model import (
    get_weight, set_weight,
)


# ═══════════════════════════════════════════════════════════════════════
#  Calibration Data Generator
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_calibration_data(
    model: nn.Module,
    tokenizer,
    num_batches: int = 4,
    batch_size: int = 1,
    max_length: int = 128,
    device: torch.device = None,
) -> Dict[int, torch.Tensor]:
    """Run model on calibration data, cache hidden states at block boundaries.

    For each block index i, cache the hidden state AFTER block i
    (i.e., the input to block i+1). Also cache the input to block 0
    (the transformer embedding output).

    Args:
        model: GPT-2 model.
        tokenizer: HuggingFace tokenizer.
        num_batches: Number of batches of calibration data.
        batch_size: Batch size.
        max_length: Max sequence length.
        device: Device.

    Returns:
        Dict mapping block_index -> tensor of hidden states.
        - cache[0]: input to block 0 (after embedding + position + dropout)
        - cache[i]: output of block i-1 (i.e., input to block i)
        - cache[n_blocks]: output of last block
        Combined over all batches.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        return {}

    texts = [t for t in dataset["text"] if len(t.strip()) > 20]
    n_blocks = len(model.transformer.h)

    # Initialize cache lists
    cache_lists: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_blocks + 1)}

    for start_idx in range(0, min(len(texts), num_batches * batch_size), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        if not batch_texts:
            break

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = encoded["input_ids"]
        if input_ids.size(1) < 2:
            continue

        # Embedding + position + dropout
        hidden_states = model.transformer.wte(input_ids)
        position_ids = torch.arange(max_length, device=device).unsqueeze(0)
        hidden_states = hidden_states + model.transformer.wpe(position_ids)
        hidden_states = model.transformer.drop(hidden_states)

        # Cache input to block 0
        cache_lists[0].append(hidden_states.cpu())

        # Run through blocks, caching outputs
        for i, block in enumerate(model.transformer.h):
            hidden_states = block(hidden_states)[0]
            cache_lists[i + 1].append(hidden_states.cpu())

    # Concatenate all batches
    cache = {}
    for i in range(n_blocks + 1):
        if cache_lists[i]:
            cache[i] = torch.cat(cache_lists[i], dim=0)
            print(f"  Cached block {i}: {cache[i].shape}")
        else:
            print(f"  WARNING: No calibration data for block {i}")
            cache[i] = torch.empty(0)

    return cache


# ═══════════════════════════════════════════════════════════════════════
#  Functional GPT-2 Block Forward with Differentiable Weights
# ═══════════════════════════════════════════════════════════════════════

def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal attention mask (1 = attend, 0 = masked)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


def gpt2_block_forward_functional(
    hidden_states: torch.Tensor,
    block: nn.Module,
    rotor_dict: Dict[str, torch.Tensor],
    quantizer: UniformQuantizer,
    apply_quantization: bool = True,
    return_debug: bool = False,
) -> torch.Tensor:
    """Forward through one GPT-2 block with differentiable rotor-applied weights.

    This is a functional reimplementation of GPT2Block.forward that uses
    externally provided rotation matrices applied to the weight matrices.
    The rotation is fully differentiable, allowing gradients to flow back
    to the rotor parameters.

    The process for each weight matrix W:
      1. Rotate: W_rot = W @ R^T
      2. Quantize (with STE): W_q = quantizer(W_rot)
      3. De-rotate: W_eff = W_q @ R
      4. Use W_eff for the matmul in the forward pass

    Args:
        hidden_states: Input tensor [batch, seq, hidden_dim].
        block: A GPT2Block module (contains ln_1, attn, ln_2, mlp).
        rotor_dict: Dict mapping layer names ('attn.c_attn', 'attn.c_proj',
            'mlp.c_fc', 'mlp.c_proj') to rotation matrices.
        quantizer: UniformQuantizer instance for weight quantization.
        apply_quantization: If True, apply quantization to rotated weights.
            If False, only apply rotation (useful for gradient propagation).
        return_debug: If True, return (output, debug_info).

    Returns:
        Output tensor [batch, seq, hidden_dim].
    """
    attn_module = block.attn
    mlp_module = block.mlp

    # ═══ Pre-LN Attention ═══════════════════════════════════════════════
    residual = hidden_states
    x = block.ln_1(hidden_states)

    # Get original weights in standard format (out_dim, in_dim)
    W_ca = get_weight(attn_module.c_attn, "conv1d")  # (3*hidden, hidden)
    W_cp = get_weight(attn_module.c_proj, "conv1d")  # (hidden, hidden)

    # Apply rotation and quantization
    R_ca = rotor_dict.get("attn.c_attn")
    R_cp = rotor_dict.get("attn.c_proj")

    if R_ca is not None:
        W_ca_rot = W_ca @ R_ca.T
        W_ca_eff = quantizer(W_ca_rot) if apply_quantization else W_ca_rot
    else:
        W_ca_eff = quantizer(W_ca) if apply_quantization else W_ca

    if R_cp is not None:
        W_cp_rot = W_cp @ R_cp.T
        W_cp_eff = quantizer(W_cp_rot) if apply_quantization else W_cp_rot
    else:
        W_cp_eff = quantizer(W_cp) if apply_quantization else W_cp

    # QKV projection
    # In Conv1D format: output = x @ W.T + bias (W is (in_dim, out_dim))
    # Our weights are in standard (out_dim, in_dim) format
    # So: output = F.linear(x, W_eff) where W_eff is (out_dim, in_dim)
    qkv = F.linear(x, W_ca_eff) + attn_module.c_attn.bias

    # Split Q, K, V
    split_size = attn_module.split_size  # hidden_dim (n_embd)
    query, key, value = qkv.split(split_size, dim=-1)

    num_heads = attn_module.num_heads
    head_dim = split_size // num_heads

    # Reshape to [batch, num_heads, seq, head_dim]
    def reshape_for_attention(x):
        batch, seq, _ = x.shape
        return x.view(batch, seq, num_heads, head_dim).transpose(1, 2)

    query = reshape_for_attention(query)
    key = reshape_for_attention(key)
    value = reshape_for_attention(value)

    # Scaled dot-product attention with causal mask
    seq_len = query.size(-2)
    causal_mask = _make_causal_mask(seq_len, hidden_states.device)

    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = attn_weights + causal_mask[:, :, :seq_len, :seq_len]
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=attn_module.attn_dropout.p if attn_module.training else 0.0)

    attn_output = torch.matmul(attn_weights, value)
    batch, seq = hidden_states.shape[:2]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, -1)

    # Output projection
    attn_output = F.linear(attn_output, W_cp_eff) + attn_module.c_proj.bias
    attn_output = attn_module.resid_dropout(attn_output)

    # Residual connection
    hidden_states = residual + attn_output

    # ═══ Pre-LN MLP ════════════════════════════════════════════════════
    residual = hidden_states
    x = block.ln_2(hidden_states)

    # Get MLP weights
    W_fc = get_weight(mlp_module.c_fc, "conv1d")    # (4*hidden, hidden)
    W_mlp_cp = get_weight(mlp_module.c_proj, "conv1d")  # (hidden, 4*hidden)

    # Apply rotation and quantization
    R_fc = rotor_dict.get("mlp.c_fc")
    R_mlp_cp = rotor_dict.get("mlp.c_proj")

    if R_fc is not None:
        W_fc_rot = W_fc @ R_fc.T
        W_fc_eff = quantizer(W_fc_rot) if apply_quantization else W_fc_rot
    else:
        W_fc_eff = quantizer(W_fc) if apply_quantization else W_fc

    if R_mlp_cp is not None:
        W_mlp_cp_rot = W_mlp_cp @ R_mlp_cp.T
        W_mlp_cp_eff = quantizer(W_mlp_cp_rot) if apply_quantization else W_mlp_cp_rot
    else:
        W_mlp_cp_eff = quantizer(W_mlp_cp) if apply_quantization else W_mlp_cp

    # MLP forward: c_fc -> gelu -> c_proj -> dropout
    fc_output = F.linear(x, W_fc_eff) + mlp_module.c_fc.bias
    fc_output = F.gelu(fc_output)
    mlp_output = F.linear(fc_output, W_mlp_cp_eff) + mlp_module.c_proj.bias
    mlp_output = mlp_module.dropout(mlp_output)

    # Residual connection
    hidden_states = residual + mlp_output

    if return_debug:
        debug = {
            "qkv_max": qkv.abs().max().item(),
            "attn_output_max": attn_output.abs().max().item(),
            "fc_output_max": fc_output.abs().max().item(),
        }
        return hidden_states, debug

    return hidden_states


# ═══════════════════════════════════════════════════════════════════════
#  Block-Wise Joint Rotor Optimization
# ═══════════════════════════════════════════════════════════════════════
class SubModelRotorOptimizer(nn.Module):
    """Optimizes rotors jointly within a sub-model (group of consecutive blocks).

    The sub-model takes hidden states at block i, runs through blocks i..i+k-1,
    and minimizes the MSE between:
      - Output with rotated + quantized weights
      - Reference output (from FP16 forward pass, cached)

    All rotors in the sub-model are optimized JOINTLY via gradient descent.
    """

    def __init__(
        self,
        model: nn.Module,
        start_block: int,
        end_block: int,
        n_bits: int = 4,
        n_steps: int = 50,
        lr: float = 0.05,
        device: torch.device = None,
    ):
        super().__init__()
        """
        Args:
            model: The original (unquantized) GPT-2 model.
            start_block: First block index (inclusive).
            end_block: Last block index (exclusive), i.e., blocks [start, end).
            n_bits: Quantization bit-width.
            n_steps: Optimization steps.
            lr: Learning rate for rotor angle optimization.
            device: Device.
        """
        self.model = model
        self.start_block = start_block
        self.end_block = end_block
        self.n_bits = n_bits
        self.n_steps = n_steps
        self.lr = lr
        self.device = device or next(model.parameters()).device

        self.blocks = model.transformer.h[start_block:end_block]
        self.quantizer = UniformQuantizer(n_bits, symmetric=True, per_channel=True)

        # Build rotor parameters for all layers in this sub-model
        # NOTE: nn.ParameterDict cannot handle names with dots, so we
        # use a regular dict and register params with safe names.
        self.rotor_params: Dict[str, nn.Parameter] = {}
        self.rotor_dims: Dict[str, int] = {}
        self._init_rotors()

    def _init_rotors(self):
        """Initialize rotor angle parameters for all layers in the sub-model."""
        for bidx in range(self.start_block, self.end_block):
            block = self.model.transformer.h[bidx]
            for layer_type in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
                full_name = f"h.{bidx}.{layer_type}"
                safe_name = full_name.replace(".", "_")
                # Get dimension
                if layer_type.startswith("attn"):
                    subname = layer_type.split(".")[1]
                    if subname == "c_attn":
                        W = get_weight(block.attn.c_attn, "conv1d")
                    elif subname == "c_proj":
                        W = get_weight(block.attn.c_proj, "conv1d")
                    else:
                        raise ValueError(f"Unknown attn layer: {subname}")
                else:  # mlp
                    subname = layer_type.split(".")[1]
                    if subname == "c_fc":
                        W = get_weight(block.mlp.c_fc, "conv1d")
                    elif subname == "c_proj":
                        W = get_weight(block.mlp.c_proj, "conv1d")
                    else:
                        raise ValueError(f"Unknown mlp layer: {subname}")

                dim = W.shape[1]
                n_rotors = max(dim // 2, 1)
                angles = nn.Parameter(torch.zeros(n_rotors, device=self.device))
                self.rotor_params[full_name] = angles
                self.register_parameter(f"rotor_{safe_name}", angles)
                self.rotor_dims[full_name] = dim

    def _build_rotor_matrices(self) -> Dict[str, torch.Tensor]:
        """Build rotation matrices from angle parameters for all layers."""
        rotors = {}
        for name, angles in self.rotor_params.items():
            dim = self.rotor_dims[name]
            rotors[name] = block_diag_rotor_matrix(angles, dim)
        return rotors

    def _forward_sub_model(
        self,
        hidden_states: torch.Tensor,
        apply_quantization: bool = True,
    ) -> torch.Tensor:
        """Run the sub-model forward with current rotors.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim].
            apply_quantization: If True, quantize rotated weights.

        Returns:
            Output hidden states after all blocks in the sub-model.
        """
        full_rotors = self._build_rotor_matrices()

        for i, block in enumerate(self.blocks):
            block_idx = self.start_block + i
            prefix = f"h.{block_idx}"

            # Build per-block rotor dict
            block_rotors = {}
            for layer_type in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
                key = f"{prefix}.{layer_type}"
                if key in full_rotors:
                    block_rotors[layer_type] = full_rotors[key]

            hidden_states = gpt2_block_forward_functional(
                hidden_states, block, block_rotors, self.quantizer,
                apply_quantization=apply_quantization,
            )

        return hidden_states

    def optimize(
        self,
        calibration_input: torch.Tensor,
        calibration_target: torch.Tensor,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Optimize all rotors in this sub-model jointly.

        Args:
            calibration_input: Hidden states at the start of the sub-model
                [batch, seq, hidden_dim].
            calibration_target: Reference hidden states at the end of the sub-model
                (from FP16 forward pass) [batch, seq, hidden_dim].

        Returns:
            Dict of optimized rotation matrices (full_name -> R matrix).
        """
        calibration_input = calibration_input.to(self.device)
        calibration_target = calibration_target.to(self.device)

        # Initialize RTN baseline loss (for reporting improvement)
        with torch.no_grad():
            rtn_output = self._forward_sub_model(
                calibration_input, apply_quantization=True
            )
            rtn_mse = F.mse_loss(rtn_output, calibration_target).item()
            rten_mse_str = f"{rtn_mse:.6f}"
            if verbose:
                print(f"      RTN baseline MSE: {rten_mse_str}")

        # Optimize
        optimizer = torch.optim.Adam(self.rotor_params.values(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_steps
        )

        best_loss = float("inf")
        best_angles = {}

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Forward with quantization
            output = self._forward_sub_model(
                calibration_input, apply_quantization=True
            )
            loss = F.mse_loss(output, calibration_target)
            loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.rotor_params.values(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                for name in self.rotor_params:
                    best_angles[name] = self.rotor_params[name].detach().clone()

            if verbose and (step + 1) % 10 == 0:
                print(f"      Step {step+1}/{self.n_steps}: MSE = {current_loss:.6f}")

        # Restore best angles
        with torch.no_grad():
            for name in self.rotor_params:
                self.rotor_params[name].copy_(best_angles[name])

        # Build final rotation matrices
        final_rotors = self._build_rotor_matrices()

        # Compute final MSE and improvement
        with torch.no_grad():
            final_output = self._forward_sub_model(
                calibration_input, apply_quantization=True
            )
            final_mse = F.mse_loss(final_output, calibration_target).item()
            impr = (rtn_mse - final_mse) / rtn_mse * 100 if rtn_mse > 0 else 0

        if verbose:
            print(f"      Final MSE: {final_mse:.6f} ({impr:+.1f}% vs RTN)")

        return final_rotors, {"rtn_mse": rtn_mse, "final_mse": final_mse, "improvement_pct": impr}


# ═══════════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════════

def optimize_ensemble_rotors(
    model: nn.Module,
    tokenizer,
    n_bits: int = 4,
    n_steps_per_submodel: int = 50,
    submodel_size: int = 2,
    n_calibration_batches: int = 4,
    calibration_batch_size: int = 2,
    max_length: int = 128,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Optimize rotors using rotation-aware ensemble quantization.

    Divides the model into sub-models of `submodel_size` consecutive blocks,
    generates calibration data, and jointly optimizes all rotors within each
    sub-model to minimize output reconstruction error.

    Args:
        model: The original (unquantized) GPT-2 model.
        tokenizer: HuggingFace tokenizer.
        n_bits: Quantization bit-width.
        n_steps_per_submodel: Optimization steps per sub-model.
        submodel_size: Number of blocks per sub-model.
        n_calibration_batches: Number of batches for calibration data.
        calibration_batch_size: Batch size for calibration data.
        max_length: Max sequence length for calibration data.
        verbose: Print progress.

    Returns:
        (model_with_absorbed_rotors, info_dict).
    """
    device = next(model.parameters()).device
    n_blocks = len(model.transformer.h)
    all_info = {}

    # Step 2: Process sub-models
    n_submodels = (n_blocks + submodel_size - 1) // submodel_size
    print(f"\n  Optimizing {n_submodels} sub-models ({submodel_size} blocks each)...")

    # Build a list of (start_block, end_block) pairs
    submodel_ranges = []
    for sm_idx in range(n_submodels):
        start = sm_idx * submodel_size
        end = min(start + submodel_size, n_blocks)
        submodel_ranges.append((start, end))

    # For each sub-model, optimize rotors and absorb them
    for sm_idx, (start_block, end_block) in enumerate(submodel_ranges):
        print(f"\n  [{sm_idx+1}/{n_submodels}] Blocks {start_block}-{end_block-1}")

        # Refresh calibration data from the current model state.
        # This keeps the optimization target aligned with any rotors already
        # absorbed into earlier sub-models.
        print(f"      Generating calibration data ({n_calibration_batches} batches)...")
        t_cache = time.time()
        cache = generate_calibration_data(
            model, tokenizer,
            num_batches=n_calibration_batches,
            batch_size=calibration_batch_size,
            max_length=max_length,
            device=device,
        )
        print(f"      Calibration data ready in {time.time() - t_cache:.1f}s")

        if not cache:
            print("      ERROR: No calibration data generated!")
            return model, {"error": "no calibration data"}

        cal_input = cache[start_block]
        cal_target = cache[end_block]

        if cal_input.numel() == 0 or cal_target.numel() == 0:
            print(f"      WARNING: Empty calibration data for blocks {start_block}-{end_block}")
            continue

        print(f"      Input: {cal_input.shape}, Target: {cal_target.shape}")

        # Create optimizer
        optimizer = SubModelRotorOptimizer(
            model, start_block, end_block,
            n_bits=n_bits, n_steps=n_steps_per_submodel,
            lr=0.05, device=device,
        )

        print(f"      Rotors: {len(optimizer.rotor_params)} layers")
        for name, angles in optimizer.rotor_params.items():
            dim = optimizer.rotor_dims[name]
            print(f"        {name}: {dim}D ({angles.numel()} angles)")

        # Optimize
        t0 = time.time()
        rotors, stats = optimizer.optimize(
            cal_input, cal_target, verbose=verbose,
        )
        elapsed = time.time() - t0

        print(f"      Optimized in {elapsed:.1f}s, MSE: RTN={stats['rtn_mse']:.6f} -> "
              f"Ensemble={stats['final_mse']:.6f} ({stats['improvement_pct']:+.1f}%)")

        # Absorb rotors into weights
        print(f"      Absorbing rotors...")
        for layer_name, R in rotors.items():
            prefix, layer_type = layer_name.rsplit(".", 1)
            block_idx = int(prefix.split(".")[1])

            # Find the module
            block = model.transformer.h[block_idx]
            if "attn" in layer_name:
                if "c_attn" in layer_name:
                    mod, ltype = block.attn.c_attn, "conv1d"
                else:
                    mod, ltype = block.attn.c_proj, "conv1d"
            else:
                if "c_fc" in layer_name:
                    mod, ltype = block.mlp.c_fc, "conv1d"
                else:
                    mod, ltype = block.mlp.c_proj, "conv1d"

            W = get_weight(mod, ltype)
            W_rotated = W @ R.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_rotated) @ R
            set_weight(mod, ltype, W_q)

        all_info[f"submodel_{sm_idx}"] = {
            "blocks": (start_block, end_block),
            "rtn_mse": stats["rtn_mse"],
            "final_mse": stats["final_mse"],
            "improvement_pct": stats["improvement_pct"],
            "time": elapsed,
            "n_rotors": len(rotors),
        }

    # Summary
    print(f"\n  Ensemble optimization complete:")
    improvements = [v["improvement_pct"] for v in all_info.values()
                    if "improvement_pct" in v]
    if improvements:
        print(f"    Mean improvement per sub-model: {sum(improvements)/len(improvements):.1f}%")
        print(f"    Sub-models improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    return model, all_info
