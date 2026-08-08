"""SwiGLU MLP task."""

TASK = {
    "title": "SwiGLU MLP",
    "difficulty": "Medium",
    "function_name": "SwiGLUMLP",
    "hint": "Three nn.Linear layers: gate_proj(d, d_ff), up_proj(d, d_ff), down_proj(d_ff, d). forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x)). SiLU(x) = x * sigmoid(x).",
    "tests": [
        {
            "name": "Parameter shapes",
            "code": """
import torch, torch.nn as nn
mlp = {fn}(d_model=64, d_ff=128)
assert isinstance(mlp, nn.Module), 'Must inherit from nn.Module'
assert hasattr(mlp, 'gate_proj'), 'Need self.gate_proj'
assert hasattr(mlp, 'up_proj'), 'Need self.up_proj'
assert hasattr(mlp, 'down_proj'), 'Need self.down_proj'
assert mlp.gate_proj.weight.shape == (128, 64), f'gate_proj shape: {mlp.gate_proj.weight.shape}'
assert mlp.up_proj.weight.shape == (128, 64), f'up_proj shape: {mlp.up_proj.weight.shape}'
assert mlp.down_proj.weight.shape == (64, 128), f'down_proj shape: {mlp.down_proj.weight.shape}'
""",
        },
        {
            "name": "Forward output shape",
            "code": """
import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(2, 8, 32)
out = mlp(x)
assert out.shape == (2, 8, 32), f'Output shape: {out.shape}'
""",
        },
        {
            "name": "Numerical correctness",
            "code": """
import torch, torch.nn.functional as F
torch.manual_seed(0)
mlp = {fn}(d_model=16, d_ff=32)
x = torch.randn(1, 4, 16)
out = mlp(x)
gate = mlp.gate_proj(x)
up = mlp.up_proj(x)
ref = mlp.down_proj(F.silu(gate) * up)
assert torch.allclose(out, ref, atol=1e-5), 'Output != down(silu(gate(x)) * up(x))'
""",
        },
        {
            "name": "2-D input",
            "code": """
import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(4, 32)
out = mlp(x)
assert out.shape == (4, 32), f'2-D output shape: {out.shape}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
mlp = {fn}(d_model=32, d_ff=64)
x = torch.randn(2, 4, 32, requires_grad=True)
mlp(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in mlp.parameters())
n_grad = sum(1 for p in mlp.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
    ],
}
