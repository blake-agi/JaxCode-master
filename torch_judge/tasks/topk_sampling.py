"""Top-k / Top-p Sampling task."""

TASK = {
    "title": "Top-k / Top-p Sampling",
    "difficulty": "Medium",
    "function_name": "sample_top_k_top_p",
    "hint": "Apply temperature first. For top-k: set logits below the k-th largest to -inf. For top-p: sort, compute cumsum of probs, mask where cumsum > p. Then sample from softmax.",
    "tests": [
        {
            "name": "top_k=1 always returns argmax",
            "code": "\nimport torch\ntorch.manual_seed(0)\nlogits = torch.tensor([1.0, 5.0, 2.0, 0.5])\nfor _ in range(10):\n    assert {fn}(logits.clone(), top_k=1) == 1, 'top_k=1 should return argmax'\n"
        },
        {
            "name": "Low temperature concentrates",
            "code": "\nimport torch\ntorch.manual_seed(42)\nlogits = torch.tensor([1.0, 3.0, 2.0])\ncounts = [0, 0, 0]\nfor _ in range(100):\n    counts[{fn}(logits.clone(), temperature=0.01)] += 1\nassert counts[1] > 90, f'Low temp should pick argmax, got {counts}'\n"
        },
        {
            "name": "All tokens reachable (no filtering)",
            "code": "\nimport torch\nlogits = torch.zeros(5)\nseen = set()\nfor i in range(200):\n    torch.manual_seed(i)\n    seen.add({fn}(logits.clone()))\nassert len(seen) == 5, f'Only saw {seen}'\n"
        },
        {
            "name": "Returns valid index",
            "code": "\nimport torch\ntorch.manual_seed(0)\nV = 100\nlogits = torch.randn(V)\nfor _ in range(20):\n    t = {fn}(logits.clone(), top_k=10, top_p=0.9)\n    assert 0 <= t < V, f'Token {t} out of range'\n"
        }
    ]
}
