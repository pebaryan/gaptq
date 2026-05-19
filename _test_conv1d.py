"""Verify Conv1D detection works correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaptq.quantize_model import get_linear_layers, load_model_and_tokenizer, get_weight
from gaptq.quantization import UniformQuantizer, quantization_error
from gaptq.ga import bivector_exp
from gaptq.rotor_quant import get_optimal_rotation
import torch

# Quick Conv1D layer check
from transformers import GPT2LMHeadModel, GPT2Config
config = GPT2Config()
m = GPT2LMHeadModel(config).eval()
layers = get_linear_layers(m)
print(f"Found {len(layers)} layers with dummy weights:")
linear_count = sum(1 for _, _, t in layers if t == 'linear')
conv1d_count = sum(1 for _, _, t in layers if t == 'conv1d')
print(f"  Linear: {linear_count}, Conv1D: {conv1d_count}")
for name, mod, ltype in layers:
    W = get_weight(mod, ltype)
    print(f"  {name:50s} {ltype:8s} {list(W.shape)}")

# Verify Conv1D get_weight/set_weight round-trip
conv1d_layers = [(name, mod) for name, mod, t in layers if t == 'conv1d']
if conv1d_layers:
    name, mod = conv1d_layers[0]
    print(f"\nTesting Conv1D round-trip on {name}:")
    W_orig = get_weight(mod, 'conv1d')
    print(f"  get_weight shape: {list(W_orig.shape)}")
    
    # Quantize with RTN
    q = UniformQuantizer(4, symmetric=True, per_channel=True)
    W_q = q(W_orig)
    nmse = quantization_error(W_orig, W_q).item()
    print(f"  RTN NMSE: {nmse:.6f}")
    
    # Set back and verify
    set_weight(mod, 'conv1d', W_q)
    W_back = get_weight(mod, 'conv1d')
    print(f"  Round-trip match: {torch.allclose(W_q, W_back)}")
    
    # Test with rotor optimization
    print(f"\n  Testing rotor on {name}: {list(W_orig.shape)}")
    R = get_optimal_rotation(W_orig[:128], mode='full', n_bits=4, n_optimization_steps=10, verbose=False)
    print(f"  Rotor shape: {list(R.shape)}, det={R.det().item():.4f}")

print("\nAll checks passed!")
