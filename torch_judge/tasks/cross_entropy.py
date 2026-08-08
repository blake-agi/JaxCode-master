"""Cross-Entropy Loss task."""

TASK = {
    "title": "Cross-Entropy Loss",
    "difficulty": "Easy",
    "function_name": "cross_entropy_loss",
    "hint": "log_probs = logits - logsumexp(logits, dim=-1, keepdim=True). Loss = -log_probs[arange(B), targets].mean(). Subtract max for stability (logsumexp handles this).",
    "tests": [
        {
            "name": "Matches F.cross_entropy",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.randn(4, 10)\ntargets = torch.randint(0, 10, (4,))\nout = {fn}(logits, targets)\nref = torch.nn.functional.cross_entropy(logits, targets)\nassert torch.allclose(out, ref, atol=1e-5), f'Mismatch: {out.item():.4f} vs {ref.item():.4f}'\n"
        },
        {
            "name": "Numerical stability",
            "code": "\nimport torch\nlogits = torch.tensor([[1000., 0., 0.], [0., 1000., 0.]])\ntargets = torch.tensor([0, 1])\nout = {fn}(logits, targets)\nassert not torch.isnan(out), 'NaN with large logits'\nassert not torch.isinf(out), 'Inf with large logits'\nassert out.item() < 0.01, 'Should be ~0 for confident correct predictions'\n"
        },
        {
            "name": "Scalar output",
            "code": "\nimport torch\nout = {fn}(torch.randn(8, 5), torch.randint(0, 5, (8,)))\nassert out.dim() == 0, 'Loss must be a scalar'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nlogits = torch.randn(8, 5, requires_grad=True)\ntargets = torch.randint(0, 5, (8,))\n{fn}(logits, targets).backward()\nassert logits.grad is not None, 'logits.grad is None'\n"
        }
    ]
}
