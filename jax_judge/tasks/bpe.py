"""Byte-Pair Encoding — train merges by frequency, then apply them in order."""

TASK = {
    "title": "Byte-Pair Encoding (BPE)",
    "category": "Inference & Decoding",
    "order": 4,
    "number": "35",
    "difficulty": "Hard",
    "function_name": "SimpleBPE",
    "hint": (
        "Represent each word as a tuple of symbols plus an end-of-word marker "
        "'</w>', kept in a {symbols: frequency} dict. Each round, count every "
        "adjacent pair weighted by word frequency, take the most frequent, "
        "record it, and rewrite every word with that pair fused. encode() then "
        "replays the recorded merges IN ORDER — the order is the model, so "
        "applying them by frequency instead of by training order gives different "
        "tokens."
    ),
    "description": r"""
Implement **Byte-Pair Encoding**: learn a merge table from a corpus, then use
it to tokenize new text.

### Signature
```python
class SimpleBPE:
    def __init__(self): ...                        # self.merges = []
    def train(self, corpus, num_merges): ...       # fills self.merges
    def encode(self, text): ...                    # -> list[str]
```

### The algorithm
1. Split every word into characters and append `'</w>'` as an end-of-word marker
2. Count each word's frequency in the corpus
3. Repeat `num_merges` times:
   - count every adjacent symbol pair, weighted by word frequency
   - take the **most frequent** pair, append it to `self.merges`
   - rewrite every word with that pair fused into one symbol
4. `encode` splits on whitespace, appends `'</w>'`, and replays `self.merges`
   **in training order**

### Rules
- Pure Python — no JAX needed here, and no `tokenizers`/`sentencepiece`
- Stop early if there are no pairs left
- `self.merges` is a list of `(a, b)` tuples in the order they were learned

### Why the merge ORDER is the whole model
The merges are not a set, they are a **sequence**. `('e','s')` learned before
`('es','t')` is what lets `est` form at all — apply them in a different order
and the second merge never matches. This is why a BPE tokenizer ships its merge
list as an ordered file, and why you cannot add a merge in the middle without
retraining.

### Why `'</w>'`
Without an end-of-word marker, `"est"` in *estimate* and `"est"` in *fastest*
are the same symbol, and the tokenizer cannot tell a prefix from a suffix. The
marker makes word-final position part of the token's identity.

### What BPE buys you
It sits between characters (no unknown tokens, but very long sequences) and
whole words (short sequences, but a huge vocabulary and an `<UNK>` problem for
anything unseen). BPE gets an open vocabulary — *any* string is encodable —
with sequences only a few times longer than word-level. The cost is that token
boundaries are a statistical artifact of the training corpus, which is exactly
why models are bad at character-level tasks like counting letters.
""",
    "stub": '''class SimpleBPE:
    """Byte-pair encoding: learn merges, then apply them."""

    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        """Learn `num_merges` merges from `corpus` (a list of words).

        Fills self.merges with (a, b) tuples in the order learned.
        """
        pass  # Replace this

    def encode(self, text):
        """Tokenize `text` by replaying self.merges in order. -> list[str]"""
        pass  # Replace this
''',
    "solution": '''class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        # Each word becomes a tuple of symbols + end-of-word marker.
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ("</w>",)
            vocab[symbols] = vocab.get(symbols, 0) + 1

        self.merges = []
        for _ in range(num_merges):
            # Count adjacent pairs, weighted by how often the word occurs.
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)

            # Rewrite every word with `best` fused into a single symbol.
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

    def encode(self, text):
        all_tokens = []
        for word in text.split():
            symbols = list(word) + ["</w>"]
            # Replay merges IN TRAINING ORDER — the order is the model.
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols = symbols[:i] + [a + b] + symbols[i + 2:]
                    else:
                        i += 1
            all_tokens.extend(symbols)
        return all_tokens
''',
    "demo": '''bpe = SimpleBPE()
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3

bpe.train(corpus, num_merges=10)
print("learned merges, in order:")
for i, m in enumerate(bpe.merges):
    print(f"  {i}: {m}")

print("\\nencode('newest'):", bpe.encode("newest"))
print("encode('lowest'):", bpe.encode("lowest"), "  <- never seen, still encodable")
''',
    "tests": [
        {
            "name": "Learns the most frequent pair first",
            "code": """
bpe = {fn}()
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
bpe.train(corpus, num_merges=1)

assert len(bpe.merges) == 1, f'Expected 1 merge, got {len(bpe.merges)}'
# ('e','s') appears in newest(6) + widest(3) = 9, more than any other pair.
assert bpe.merges[0] == ("e", "s"), (
    f"First merge should be ('e', 's') with count 9, got {bpe.merges[0]}"
)
""",
        },
        {
            "name": "Merges accumulate in order",
            "code": """
bpe = {fn}()
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
bpe.train(corpus, num_merges=4)

assert len(bpe.merges) == 4, f'Expected 4 merges, got {len(bpe.merges)}'
assert bpe.merges[0] == ("e", "s"), f'{bpe.merges[0]}'
assert bpe.merges[1] == ("es", "t"), (
    f"Second merge should fuse the previous result: ('es','t'), got {bpe.merges[1]}"
)
assert bpe.merges[2] == ("est", "</w>"), f'{bpe.merges[2]}'
assert all(isinstance(m, tuple) and len(m) == 2 for m in bpe.merges), (
    'merges must be a list of (a, b) tuples'
)
""",
        },
        {
            "name": "encode replays the merges",
            "code": """
bpe = {fn}()
bpe.train(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3, num_merges=4)

toks = bpe.encode("newest")
assert isinstance(toks, list), f'encode must return a list, got {type(toks)}'
assert "est</w>" in toks, (
    f"After 4 merges 'est</w>' should be one token, got {toks}"
)
assert "".join(toks) == "newest</w>", f'Tokens must reconstruct the word: {toks}'
""",
        },
        {
            "name": "Untrained encoder splits into characters",
            "code": """
bpe = {fn}()
assert bpe.merges == [], 'A fresh SimpleBPE should have no merges'

toks = bpe.encode("hi there")
assert toks == ["h", "i", "</w>", "t", "h", "e", "r", "e", "</w>"], (
    f'With no merges, encode should give characters plus </w> markers, got {toks}'
)
""",
        },
        {
            "name": "Open vocabulary: unseen words still encode",
            "code": """
bpe = {fn}()
bpe.train(["low"] * 5 + ["newest"] * 6, num_merges=5)

toks = bpe.encode("zzz")
assert "".join(toks) == "zzz</w>", f'Unseen word must still encode: {toks}'
assert all(isinstance(t, str) for t in toks), 'Tokens must be strings'

multi = bpe.encode("low newest")
assert "".join(multi) == "low</w>newest</w>", f'{multi}'
assert multi.count("</w>") >= 2, 'Each word needs its own end-of-word marker'
""",
        },
        {
            "name": "Stops early when no pairs remain",
            "code": """
bpe = {fn}()
# Single-character words: 'a' + '</w>' gives exactly one pair, then nothing.
bpe.train(["a"], num_merges=100)
assert len(bpe.merges) <= 1, (
    f'Should stop when no pairs are left, got {len(bpe.merges)} merges'
)

empty = {fn}()
empty.train([], num_merges=5)
assert empty.merges == [], 'An empty corpus should learn no merges'
""",
        },
        {
            "name": "Frequency weighting uses word counts",
            "code": """
bpe = {fn}()
# 'ab' appears in one word repeated 100x; 'xy' in three distinct words once each.
bpe.train(["ab"] * 100 + ["xy", "xyz", "xyw"], num_merges=1)
assert bpe.merges[0] == ("a", "b"), (
    f"Pair counting must weight by word FREQUENCY (ab: 100 vs xy: 3), got {bpe.merges[0]}"
)
""",
        },
    ],
}
