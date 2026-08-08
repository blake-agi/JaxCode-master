"""Mixture of Experts (MoE) task."""

TASK = {
    "title": "Mixture of Experts (MoE)",
    "difficulty": "Hard",
    "function_name": "MixtureOfExperts",
    "hint": "Router: Linear(d, num_experts) -> topk -> softmax. Each expert: Linear->ReLU->Linear. Weighted sum of top-k expert outputs per token.",
    "tests": [
        {
            "name": "Output shape",
            "code": "\nimport torch, torch.nn as nn\nmoe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)\nassert isinstance(moe, nn.Module)\nout = moe(torch.randn(2, 8, 32))\nassert out.shape == (2, 8, 32), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Has router and experts",
            "code": "\nimport torch, torch.nn as nn\nmoe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)\nassert hasattr(moe, 'router'), 'Need self.router'\nassert hasattr(moe, 'experts'), 'Need self.experts'\nassert len(moe.experts) == 4, f'Expected 4 experts, got {len(moe.experts)}'\n"
        },
        {
            "name": "Router logits shape",
            "code": "\nimport torch\nmoe = {fn}(d_model=16, d_ff=32, num_experts=8, top_k=2)\nlogits = moe.router(torch.randn(4, 16))\nassert logits.shape == (4, 8), f'Router output: {logits.shape}'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nmoe = {fn}(d_model=16, d_ff=32, num_experts=4, top_k=2)\nx = torch.randn(1, 4, 16, requires_grad=True)\nmoe(x).sum().backward()\nassert x.grad is not None, 'x.grad is None'\n"
        }
    ]
}
