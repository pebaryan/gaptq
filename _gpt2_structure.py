"""Inspect GPT-2's module structure to map absorption pairs."""
import sys
sys.path.insert(0, 'D:/code/gaptq')

from gaptq.quantize_model import get_linear_layers, load_model_and_tokenizer, get_weight, _is_conv1d

model, tok, dev = load_model_and_tokenizer('gpt2')

# Print full module tree
def print_tree(module, prefix=""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) or _is_conv1d(child):
            W = get_weight(child, 'linear' if isinstance(child, torch.nn.Linear) else 'conv1d')
            print(f"  {full:50s} {W.shape}")
        else:
            print(f"  {full:50s} (container)")
            print_tree(child, full)

import torch
print_tree(model)
print("\n\nAll quantizable layers:")
layers = get_linear_layers(model, exclude_embeddings=True)
for name, mod, ltype in layers:
    W = get_weight(mod, ltype)
    print(f"  {name:50s} ({ltype:8s}) {W.shape}")
