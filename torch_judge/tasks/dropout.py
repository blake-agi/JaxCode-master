"""Implement Dropout task."""

TASK = {
    "title": "Implement Dropout",
    "difficulty": "Easy",
    "function_name": "MyDropout",
    "hint": "During training: randomly zero elements with probability p, scale survivors by 1/(1-p). During eval: identity. Use torch.rand_like and compare with p.",
    "tests": [
        {
            "name": "Eval mode is identity",
            "code": "\nimport torch, torch.nn as nn\nd = {fn}(p=0.5)\nassert isinstance(d, nn.Module), 'Must inherit from nn.Module'\nd.eval()\nx = torch.randn(4, 8)\nassert torch.equal(d(x), x), 'eval mode should return input unchanged'\n"
        },
        {
            "name": "Training: zeros and scaling",
            "code": "\nimport torch\ntorch.manual_seed(42)\nd = {fn}(p=0.5)\nd.train()\nx = torch.ones(1000)\nout = d(x)\nassert (out == 0).any(), 'No zeros found during training'\nnon_zero = out[out != 0]\nassert torch.allclose(non_zero, torch.full_like(non_zero, 2.0), atol=1e-5), 'Non-zeros should be scaled by 1/(1-p)=2.0'\n"
        },
        {
            "name": "Drop rate is approximately p",
            "code": "\nimport torch\ntorch.manual_seed(0)\nd = {fn}(p=0.3)\nd.train()\nout = d(torch.ones(10000))\nfrac = (out == 0).float().mean().item()\nassert 0.25 < frac < 0.35, f'Expected ~30%% zeros, got {frac*100:.1f}%%'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nd = {fn}(p=0.5)\nd.train()\nx = torch.randn(4, 8, requires_grad=True)\nd(x).sum().backward()\nassert x.grad is not None, 'x.grad is None'\n"
        }
    ]
}
