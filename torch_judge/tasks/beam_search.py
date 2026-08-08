"""Beam Search Decoding task."""

TASK = {
    "title": "Beam Search Decoding",
    "difficulty": "Medium",
    "function_name": "beam_search",
    "hint": "Maintain beam_width hypotheses. Each step: expand each hypothesis with all tokens, keep top beam_width by total score. Stop when all beams end with eos or max_len reached.",
    "tests": [
        {
            "name": "Returns list starting with start_token",
            "code": "\nimport torch\ndef dummy(tokens): return torch.zeros(10)\nseq = {fn}(dummy, start_token=0, max_len=5, beam_width=3, eos_token=9)\nassert isinstance(seq, list), 'Must return a list'\nassert seq[0] == 0, f'First token: {seq[0]}'\n"
        },
        {
            "name": "Greedy path (beam=1)",
            "code": "\nimport torch\ndef greedy_fn(tokens):\n    lp = torch.full((5,), -10.0)\n    lp[min(len(tokens), 4)] = 0.0\n    return lp\nseq = {fn}(greedy_fn, start_token=0, max_len=5, beam_width=1, eos_token=4)\nassert seq == [0, 1, 2, 3, 4], f'Greedy: {seq}'\n"
        },
        {
            "name": "Beam finds better path than greedy",
            "code": "\nimport torch\ndef tricky(tokens):\n    lp = torch.full((6,), -100.0)\n    if len(tokens) == 1:\n        lp[1] = -1.0; lp[2] = -0.5\n    elif tokens[-1] == 1:\n        lp[5] = 0.0\n    elif tokens[-1] == 2:\n        lp[5] = -10.0\n    else:\n        lp[5] = 0.0\n    return lp\nseq = {fn}(tricky, start_token=0, max_len=5, beam_width=2, eos_token=5)\nassert seq == [0, 1, 5], f'Beam should find [0,1,5], got {seq}'\n"
        },
        {
            "name": "Stops at eos",
            "code": "\nimport torch\ndef eos_fn(tokens):\n    lp = torch.zeros(4); lp[3] = 10.0; return lp\nseq = {fn}(eos_fn, start_token=0, max_len=100, beam_width=2, eos_token=3)\nassert seq[-1] == 3 and len(seq) == 2, f'Should be [0,3], got {seq}'\n"
        }
    ]
}
