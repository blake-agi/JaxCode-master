"""Byte-Pair Encoding (BPE) task."""

TASK = {
    "title": "Byte-Pair Encoding (BPE)",
    "difficulty": "Hard",
    "function_name": "SimpleBPE",
    "hint": "train: split words into chars + </w>. Iteratively find most frequent adjacent pair, merge it. encode: apply learned merges in order to split text into subwords.",
    "tests": [
        {
            "name": "Correct number of merges",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=5)\nassert len(bpe.merges) == 5, f'Expected 5 merges, got {len(bpe.merges)}'\n"
        },
        {
            "name": "Most frequent pair merged first",
            "code": "\nbpe = {fn}()\nbpe.train(['aaa', 'aaa', 'aaa', 'bbb'], num_merges=1)\nassert bpe.merges[0] == ('a', 'a'), f'First merge: {bpe.merges[0]}'\n"
        },
        {
            "name": "Encode returns list of strings",
            "code": "\nbpe = {fn}()\nbpe.train(['low', 'lower', 'lowest'] * 3, num_merges=10)\ntokens = bpe.encode('low')\nassert isinstance(tokens, list), 'encode must return a list'\nassert all(isinstance(t, str) for t in tokens), 'tokens must be strings'\nreconstructed = ''.join(t.replace('</w>', '') for t in tokens)\nassert reconstructed == 'low', f'Reconstruction: {reconstructed}'\n"
        },
        {
            "name": "More merges -> fewer tokens",
            "code": "\nbpe1 = {fn}()\nbpe1.train(['hello'] * 10, num_merges=2)\nbpe2 = {fn}()\nbpe2.train(['hello'] * 10, num_merges=10)\nassert len(bpe2.encode('hello')) <= len(bpe1.encode('hello')), 'More merges should reduce tokens'\n"
        }
    ]
}
