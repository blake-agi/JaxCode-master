"""Kaiming Initialization task."""

TASK = {
    "title": "Kaiming Initialization",
    "difficulty": "Easy",
    "function_name": "kaiming_init",
    "hint": "For fan_in mode: std = sqrt(2 / fan_in) where fan_in = weight.shape[1]. Fill with normal(0, std). Return the tensor.",
    "tests": [
        {
            "name": "Mean approximately 0",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.empty(256, 512)\n{fn}(w)\nassert abs(w.mean().item()) < 0.02, f'Mean too far from 0: {w.mean().item():.4f}'\n"
        },
        {
            "name": "Std matches sqrt(2/fan_in)",
            "code": "\nimport torch, math\ntorch.manual_seed(0)\nfan_in = 1024\nw = torch.empty(256, fan_in)\n{fn}(w)\nexpected = math.sqrt(2.0 / fan_in)\nassert abs(w.std().item() - expected) < 0.005, f'Std {w.std().item():.4f} vs expected {expected:.4f}'\n"
        },
        {
            "name": "Returns same tensor (in-place)",
            "code": "\nimport torch\nw = torch.empty(64, 32)\nout = {fn}(w)\nassert out is w, 'Should return the same tensor'\nassert out.shape == (64, 32), 'Shape should be unchanged'\n"
        },
        {
            "name": "Smaller fan_in gives larger std",
            "code": "\nimport torch\nw1 = torch.empty(64, 16)\nw2 = torch.empty(64, 256)\n{fn}(w1)\n{fn}(w2)\nassert w1.std().item() > w2.std().item(), 'Smaller fan_in should give larger std'\n"
        }
    ]
}
