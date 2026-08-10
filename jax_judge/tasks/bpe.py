"""Byte-pair encoding — train the merges, then apply them."""

TASK = {
    "title": "Byte-Pair Encoding (train and apply)",
    "category": "Inference & Decoding",
    "order": 4,
    "difficulty": "Medium",
    "function_name": "train_bpe",
    # The tests exercise both halves, so apply_merges must be pulled from the
    # notebook namespace alongside train_bpe.
    "extra_names": ["apply_merges"],
    "hint": (
        "Keep each word as a LIST of symbols, not a string — after the first "
        "merge a symbol is more than one character, and 'is this pair adjacent' "
        "is a question about symbols. One helper does the real work: given a "
        "symbol list and a pair, walk it left to right and emit the joined "
        "symbol, skipping two positions when you match. Both halves of the task "
        "call it, which is the point — training is 'find the best pair, apply "
        "it, recount from scratch', and applying is the same rewrite replayed. "
        "Recount from scratch every round: merging changes which pairs exist. "
        "And make the argmax total: max() over a dict is only as deterministic "
        "as your tie-break."
    ),
    "description": r"""
Implement **byte-pair encoding** — both halves: learn the merge list from a
corpus, and apply it to a new word.

### Signature
```python
def train_bpe(word_counts, num_merges):
    ...  # -> list[tuple[str, str]], the merges in the order they were learned

def apply_merges(word, merges):
    ...  # -> list[str], the word's symbols after replaying every merge
```

`word_counts` is `{"low": 5, "lower": 2, ...}`. A word starts as its list of
characters. `apply_merges` must be defined too — the judge tests both.

### The algorithm
1. Count every adjacent symbol pair across the corpus, weighted by word count.
2. Take the most frequent pair. **Ties break by taking the lexicographically
   smallest pair**, so the output is deterministic.
3. Record it and merge every occurrence, scanning each word left to right.
4. Repeat `num_merges` times, or stop early if no pair occurs more than once.

`apply_merges` replays the recorded merges **in learned order** — the order is
the algorithm; applying them in a different order gives different tokens.

### Rules
- Pure Python — no `jnp` needed, and no `tokenizers`/`sentencepiece`
- Deterministic: same input, same merge list, every time
- Left-to-right, non-overlapping merges (in `aaa`, merging `(a,a)` gives `aa a`)

### Why subword tokenization exists
Word-level vocabularies cannot represent anything they did not see in training —
every new name, typo or compound becomes `<UNK>`, and the information is gone.
Character-level has no `<UNK>` problem but multiplies the sequence length by the
average characters-per-token (roughly 4x on English), and attention is quadratic
in length.

BPE splits the difference: frequent words stay single tokens, rare ones
decompose into reusable pieces, and because every symbol bottoms out in the base
alphabet, **nothing is ever unrepresentable**. That is the property that
matters — an open vocabulary at a fixed model size. This exercise uses
characters as that alphabet; production tokenizers use raw *bytes*, so that even
an unseen character or a broken UTF-8 fragment still encodes.

### What it costs
Tokenization is a frozen, corpus-dependent preprocessing step, and its seams
leak into model behaviour: arithmetic is bad partly because numbers tokenize
inconsistently, character-level tasks ("how many r's in strawberry") are hard
because the model never sees characters, and languages under-represented in the
training corpus get far more tokens per word — a direct cost and context-length
penalty for those users.
""",
    "stub": '''def train_bpe(word_counts, num_merges):
    """Learn BPE merges from a corpus.

    Args:
        word_counts: {word: frequency}
        num_merges:  maximum number of merges to learn

    Returns:
        list[tuple[str, str]] — merges in the order learned.
    """
    pass  # Replace this


def apply_merges(word, merges):
    """Apply a learned merge list to a word.

    Args:
        word:   the string to tokenize
        merges: list[tuple[str, str]] from train_bpe

    Returns:
        list[str] of symbols.
    """
    pass  # Replace this
''',
    "solution": '''def _merge_pair(symbols, pair):
    """Left-to-right, non-overlapping merge of `pair` inside one symbol list."""
    out = []
    i = 0
    while i < len(symbols):
        if (
            i < len(symbols) - 1
            and symbols[i] == pair[0]
            and symbols[i + 1] == pair[1]
        ):
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(word_counts, num_merges):
    # Each word becomes a list of characters, carrying its corpus frequency.
    vocab = {word: list(word) for word in word_counts}
    merges = []

    for _ in range(num_merges):
        pair_counts = {}
        for word, symbols in vocab.items():
            freq = word_counts[word]
            for a, b in zip(symbols, symbols[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + freq

        if not pair_counts:
            break

        # max() alone is not deterministic across dict orderings, so break ties
        # on the pair itself: highest count, then lexicographically smallest.
        best = min(pair_counts, key=lambda p: (-pair_counts[p], p))
        if pair_counts[best] < 2:
            break

        merges.append(best)
        vocab = {w: _merge_pair(s, best) for w, s in vocab.items()}

    return merges


def apply_merges(word, merges):
    symbols = list(word)
    # Order matters: the merges must be replayed exactly as they were learned.
    for pair in merges:
        symbols = _merge_pair(symbols, pair)
    return symbols
''',
    "demo": '''corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

merges = train_bpe(corpus, num_merges=10)
for i, m in enumerate(merges, 1):
    print(f"{i:>2}. {m[0]!r} + {m[1]!r} -> {m[0] + m[1]!r}")

print()
for w in ("lowest", "newer", "wildest"):
    print(f"{w:>8} -> {apply_merges(w, merges)}")
''',
    "tests": [
        {
            "name": "Learns the highest-frequency pair first",
            "code": """
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges = {fn}(corpus, 1)

assert isinstance(merges, list), f'Must return a list, got {type(merges).__name__}'
assert len(merges) == 1, f'Asked for 1 merge, got {len(merges)}'
assert isinstance(merges[0], tuple) and len(merges[0]) == 2, (
    f'Each merge must be a 2-tuple of strings, got {merges[0]!r}'
)

# ('e','s') appears in newest (6) and widest (3) = 9; ('s','t') is also 9 but
# ('e','s') is lexicographically smaller, so the tie-break picks it.
assert merges[0] == ('e', 's'), (
    f"Expected ('e', 's') with count 9, got {merges[0]!r}. Counts must be "
    'weighted by word frequency, and ties broken lexicographically.'
)
""",
        },
        {
            "name": "Frequencies are weighted, not just word counts",
            "code": """
# 'ab' appears in one word but 100 times; 'cd' in three words, 1 time each.
corpus = {"ab": 100, "cd": 1, "cdx": 1, "cdy": 1}
merges = {fn}(corpus, 1)

assert merges[0] == ('a', 'b'), (
    f"Expected ('a','b') with weighted count 100, got {merges[0]!r}. "
    'Pair counts must be weighted by each word frequency, not by how many '
    'distinct words contain the pair.'
)
""",
        },
        {
            "name": "Deterministic",
            "code": """
corpus = {"aa": 3, "bb": 3, "cc": 3, "ab": 1}

runs = [{fn}(dict(corpus), 3) for _ in range(5)]
for r in runs[1:]:
    assert r == runs[0], f'Non-deterministic output: {runs[0]} vs {r}'

# All of ('a','a'), ('b','b'), ('c','c') tie at 3 -> lexicographic order.
assert runs[0][0] == ('a', 'a'), f"Tie should resolve to ('a','a'), got {runs[0][0]!r}"
""",
        },
        {
            "name": "Stops early when nothing repeats",
            "code": """
corpus = {"abc": 1, "def": 1}
merges = {fn}(corpus, 100)

assert len(merges) == 0, (
    f'No pair occurs more than once, so no merge is justified. Got {merges}'
)

corpus2 = {"aa": 2}
m2 = {fn}(corpus2, 100)
assert len(m2) == 1, f'Only one merge is possible here, got {len(m2)}: {m2}'
assert m2[0] == ('a', 'a'), f"Expected ('a','a'), got {m2[0]!r}"

# num_merges is an upper bound, never exceeded.
m3 = {fn}({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 3)
assert len(m3) <= 3, f'Asked for at most 3 merges, got {len(m3)}'
""",
        },
        {
            "name": "apply_merges replays in order",
            "code": """
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges = {fn}(corpus, 10)

out = apply_merges("newest", merges)
assert isinstance(out, list), f'apply_merges must return a list, got {type(out).__name__}'
assert all(isinstance(s, str) for s in out), f'Symbols must be strings: {out}'
assert "".join(out) == "newest", (
    f'Concatenating the symbols must rebuild the word: {out} -> {"".join(out)!r}'
)

# A training word should compress to fewer symbols than characters.
assert len(out) < len("newest"), (
    f'"newest" is in the corpus so it should merge: got {out}'
)

# An unseen word is still representable — the whole point of BPE.
unseen = apply_merges("zzz", merges)
assert "".join(unseen) == "zzz", f'Unseen word broke: {unseen}'
""",
        },
        {
            "name": "Left-to-right, non-overlapping merges",
            "code": """
# In 'aaaa', merging ('a','a') left to right gives ['aa', 'aa'] — not
# ['aa', 'a', 'a'] and not a greedy overlapping pass.
out = apply_merges("aaaa", [('a', 'a')])
assert out == ['aa', 'aa'], f"Expected ['aa','aa'], got {out}"

out3 = apply_merges("aaa", [('a', 'a')])
assert out3 == ['aa', 'a'], (
    f"Expected ['aa','a'] — the merge is non-overlapping and scans left to "
    f"right, got {out3}"
)

# Merge order is load-bearing.
a = apply_merges("abab", [('a', 'b'), ('ab', 'ab')])
assert a == ['abab'], f"Sequential merges should compose: got {a}"
b = apply_merges("abab", [('ab', 'ab'), ('a', 'b')])
assert b == ['ab', 'ab'], (
    f"Applying ('ab','ab') first matches nothing (no 'ab' symbols exist yet), "
    f"so the result differs: got {b}"
)
""",
        },
        {
            "name": "Round trip on the classic corpus",
            "code": """
corpus = {"low": 5, "lowest": 2, "newer": 6, "wider": 3, "new": 2}

for n in (5, 12):
    merges = {fn}(corpus, n)
    assert len(merges) > 0, f'{n} merges: should have learned at least one'
    assert len(set(merges)) == len(merges), f'Duplicate merges learned: {merges}'
    for word in corpus:
        toks = apply_merges(word, merges)
        assert "".join(toks) == word, f'{word!r} -> {toks} does not rebuild'
        assert len(toks) <= len(word), f'{word!r} got longer: {toks}'

# Partway through training, shared subwords are visible: 'er' recurs across
# newer/wider so it becomes one symbol before the full words do.
mid = {fn}(corpus, 5)
assert apply_merges("newer", mid) == ['new', 'er'], (
    f"After 5 merges 'newer' should split as ['new','er'], got "
    f"{apply_merges('newer', mid)}"
)
assert 'er' in apply_merges("wider", mid), (
    f"The 'er' token should be reused by 'wider' too, got {apply_merges('wider', mid)}"
)

# Given enough merges, corpus words collapse to a single token — that is the
# expected end state, not a bug.
full = {fn}(corpus, 12)
assert apply_merges("lowest", full) == ['lowest'], (
    f"With 12 merges 'lowest' should be one token, got {apply_merges('lowest', full)}"
)
""",
        },
    ],
}
