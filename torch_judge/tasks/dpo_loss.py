"""DPO (Direct Preference Optimization) Loss task."""

TASK = {
    "title": "DPO (Direct Preference Optimization) Loss",
    "difficulty": "Hard",
    "function_name": "dpo_loss",
    "hint": "L = -log(sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))). Mean over batch.",
    "tests": [
        {
            "name": "Easy pair: small loss",
            "code": "\nimport torch\nchosen = torch.tensor([0.0, 0.0])\nrejected = torch.tensor([-10.0, -10.0])\nref_c = torch.tensor([-1.0, -1.0])\nref_r = torch.tensor([-1.0, -1.0])\nloss = {fn}(chosen, rejected, ref_c, ref_r, beta=0.1)\nassert loss.dim() == 0, 'Must be scalar'\nassert loss.item() < 0.5, f'Easy pair loss too high: {loss.item():.4f}'\n"
        },
        {
            "name": "Hard pair: large loss",
            "code": "\nimport torch\nloss = {fn}(torch.tensor([-10.0]), torch.tensor([0.0]),\n            torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.1)\nassert loss.item() > 0.5, f'Hard pair loss too low: {loss.item():.4f}'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nc = torch.randn(4, requires_grad=True)\nr = torch.randn(4, requires_grad=True)\n{fn}(c, r, torch.randn(4), torch.randn(4)).backward()\nassert c.grad is not None and r.grad is not None, 'Missing gradients'\n"
        },
        {
            "name": "Mathematical correctness",
            "code": "\nimport torch\ntorch.manual_seed(0)\nc, r, rc, rr = torch.randn(3), torch.randn(3), torch.randn(3), torch.randn(3)\nbeta = 0.5\nloss = {fn}(c, r, rc, rr, beta=beta)\nref = -torch.log(torch.sigmoid(beta * ((c - rc) - (r - rr)))).mean()\nassert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Beta scaling",
            "code": "\nimport torch\ntorch.manual_seed(0)\nc, r, rc, rr = torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4)\nl1 = {fn}(c, r, rc, rr, beta=0.1)\nl2 = {fn}(c, r, rc, rr, beta=1.0)\nassert not torch.allclose(l1, l2), 'Different beta should give different loss'\n"
        }
    ]
}
