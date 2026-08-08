"""Flash Attention (Tiled) task."""

TASK = {
    "title": "Flash Attention (Tiled)",
    "difficulty": "Hard",
    "function_name": "flash_attention",
    "hint": "Process Q in blocks. For each Q-block, iterate over K/V blocks. Use online softmax: track running max and sum, rescale accumulator when max changes. output = acc / row_sum.",
    "tests": [
        {
            "name": "Matches standard attention",
            "code": "\nimport torch, math\ntorch.manual_seed(0)\nB, S, D = 2, 16, 8\nQ = torch.randn(B, S, D)\nK = torch.randn(B, S, D)\nV = torch.randn(B, S, D)\nout = {fn}(Q, K, V, block_size=4)\nscores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)\nref = torch.bmm(torch.softmax(scores, dim=-1), V)\nassert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'\n"
        },
        {
            "name": "Non-aligned block size",
            "code": "\nimport torch, math\ntorch.manual_seed(42)\nB, S, D = 1, 7, 4\nQ, K, V = torch.randn(B,S,D), torch.randn(B,S,D), torch.randn(B,S,D)\nout = {fn}(Q, K, V, block_size=3)\nscores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)\nref = torch.bmm(torch.softmax(scores, dim=-1), V)\nassert torch.allclose(out, ref, atol=1e-4), 'Mismatch with non-aligned block size'\n"
        },
        {
            "name": "Block size invariant",
            "code": "\nimport torch\ntorch.manual_seed(0)\nQ, K, V = torch.randn(1,12,8), torch.randn(1,12,8), torch.randn(1,12,8)\nout4 = {fn}(Q, K, V, block_size=4)\nout6 = {fn}(Q, K, V, block_size=6)\nassert torch.allclose(out4, out6, atol=1e-4), 'Different block sizes should give same result'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nQ = torch.randn(1, 8, 4, requires_grad=True)\nK = torch.randn(1, 8, 4, requires_grad=True)\nV = torch.randn(1, 8, 4, requires_grad=True)\n{fn}(Q, K, V, block_size=4).sum().backward()\nassert Q.grad is not None, 'Q.grad is None'\n"
        }
    ]
}
