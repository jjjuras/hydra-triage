"""Structural fingerprints of prompt text: detect template reuse without any template label.

Pipeline: tokenize → mask rare tokens (ids, names, numbers: anything the corpus has seen fewer than
`min_df` times) and collapse masked runs → word 3-shingles → MinHash signature per distinct masked
structure → LSH banding to find near-duplicate structures → union them into a *family*. Two prompts
from the same template with different content fall into one family; a light paraphrase of the
template usually does too; a prompt with a genuinely new structure does not.

Pure numpy + the standard library. Deterministic for a given seed.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[a-z0-9]+")
_MASK = 0  # one symbol for all opaque material (rare tokens, anything with a digit)
_P61 = (1 << 61) - 1
_B = np.uint64(1_000_003)


def tokenize(prompts: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Flat token codes (≥1; 0 = digit-bearing token), the row id of each token, and vocab size."""
    per_row = [_TOKEN.findall(s.lower()) for s in prompts]
    lens = np.fromiter((len(t) for t in per_row), dtype=np.int64, count=len(per_row))
    toks = np.array([t for row in per_row for t in row], dtype=object)
    codes, uniques = pd.factorize(toks)
    codes = codes.astype(np.int64) + 1
    has_digit = pd.Series(uniques, dtype=object).str.contains(r"\d", regex=True).to_numpy()
    codes[has_digit[codes - 1]] = 0
    return codes, np.repeat(np.arange(len(prompts)), lens), len(uniques) + 1


def mask_and_collapse(prompts: list[str], min_df: int) -> tuple[np.ndarray, np.ndarray]:
    """Masked, run-collapsed code sequences for all prompts, flattened: (codes, row ids).
    Rare tokens (corpus count < min_df) and digit-bearing tokens become one MASK symbol. A known word
    standing alone between two masks ("... 8812 urgent 4471 ...") is content too, and is masked with
    its neighbours. Consecutive masks collapse to one, so content of any length looks the same."""
    flat, rows, n_vocab = tokenize(prompts)
    counts = np.bincount(flat, minlength=n_vocab)
    masked = np.where(counts[flat] < min_df, _MASK, flat)
    masked[flat == 0] = _MASK
    is_mask = masked == _MASK
    same_prev = np.concatenate([[False], rows[1:] == rows[:-1]])
    same_next = np.concatenate([rows[:-1] == rows[1:], [False]])
    prev_mask = np.concatenate([[False], is_mask[:-1]]) & same_prev
    next_mask = np.concatenate([is_mask[1:], [False]]) & same_next
    masked[~is_mask & prev_mask & next_mask] = _MASK
    is_mask = masked == _MASK
    prev_mask = np.concatenate([[False], is_mask[:-1]]) & same_prev
    keep = ~(is_mask & prev_mask)
    return masked[keep], rows[keep]


def row_keys(codes: np.ndarray, rows: np.ndarray, n_rows: int) -> np.ndarray:
    """Order-sensitive 64-bit hash of each row's code sequence (exact structure key)."""
    starts = np.searchsorted(rows, np.arange(n_rows))
    pos = np.arange(len(codes)) - starts[rows]
    term = (codes.astype(np.uint64) + np.uint64(3)) * (_B ** pos.astype(np.uint64))
    key = np.zeros(n_rows, dtype=np.uint64)
    np.add.at(key, rows, term)
    return key


def shingle_hashes(codes: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """One 3-shingle hash per token position: (c_i, c_i+1, c_i+2) with an end-of-row sentinel where the
    row ends, so every row, however short, has as many shingles as tokens."""
    c = codes.astype(np.uint64) + np.uint64(3)
    sent = np.uint64(1)
    nxt1 = np.concatenate([c[1:], [sent]])
    nxt2 = np.concatenate([c[2:], [sent, sent]])
    same1 = np.concatenate([rows[1:] == rows[:-1], [False]])
    same2 = np.concatenate([rows[2:] == rows[:-2], [False, False]])
    return (c * _B + np.where(same1, nxt1, sent)) * _B + np.where(same2, nxt2, sent)


def minhash(hashes: np.ndarray, starts: np.ndarray, n_hashes: int, seed: int) -> np.ndarray:
    """(n_rows, n_hashes) signatures from a flat array of shingle hashes grouped by row (row i owns
    hashes[starts[i]:starts[i+1]]). Permutations are affine maps mod 2^61-1. Duplicate shingles within a
    row do not change the minimum, so no per-row dedup is needed."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P61, size=n_hashes, dtype=np.uint64)
    b = rng.integers(0, _P61, size=n_hashes, dtype=np.uint64)
    h = hashes % np.uint64(_P61)
    sig = np.empty((len(starts), n_hashes), dtype=np.uint64)
    for j in range(n_hashes):
        sig[:, j] = np.minimum.reduceat((a[j] * h + b[j]) % np.uint64(_P61), starts)
    return sig


class _UF:
    def __init__(self, n: int) -> None:
        self.p = np.arange(n)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def structure_families(
    prompts: list[str],
    *,
    min_df: int = 5,
    n_hashes: int = 48,
    band_rows: int = 4,
    min_sim: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """Family id per prompt. Identical masked structures share an id; a structure joins an existing
    family only if its estimated Jaccard (3-shingles, MinHash) to that family's root structure is
    ≥ min_sim. Comparing to the root, not to the nearest member, stops long chains of one-word
    edits from merging unrelated structures into one blob."""
    n = len(prompts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    codes, rows = mask_and_collapse(prompts, min_df)
    keys = row_keys(codes, rows, n)
    uniq, first, key_idx = np.unique(keys, return_index=True, return_inverse=True)
    rep = np.isin(rows, first)  # one representative row per distinct structure
    rep_rows = rows[rep]
    starts = np.flatnonzero(np.concatenate([[True], rep_rows[1:] != rep_rows[:-1]]))
    sig = minhash(shingle_hashes(codes[rep], rep_rows), starts, n_hashes, seed)
    # rows of `sig` are in row order of the representatives; map back to key order
    order = np.argsort(first)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    sig = sig[inv]
    uf = _UF(len(uniq))
    for bnd in range(n_hashes // band_rows):
        band = (
            np.ascontiguousarray(sig[:, bnd * band_rows : (bnd + 1) * band_rows])
            .view(f"V{8 * band_rows}")
            .ravel()
        )
        _, grp, cnt = np.unique(band, return_inverse=True, return_counts=True)
        multi = np.flatnonzero(cnt[grp] > 1)
        if not len(multi):
            continue
        multi = multi[np.argsort(grp[multi], kind="stable")]
        bounds = np.flatnonzero(np.concatenate([[True], grp[multi][1:] != grp[multi][:-1], [True]]))
        for lo, hi in itertools.pairwise(bounds):
            members = multi[lo:hi]
            root = uf.find(int(members[0]))
            close = members[(sig[members] == sig[root]).mean(axis=1) >= min_sim]
            for m in close:
                uf.union(root, int(m))
    roots = np.array([uf.find(i) for i in range(len(uniq))])
    return roots[key_idx]


def sample_prompts(requests: pd.DataFrame, per_account: int, seed: int) -> pd.DataFrame:
    """Up to `per_account` hour-buckets per account (all of them for small accounts), deterministic."""
    rng = np.random.default_rng(seed)
    idx = requests.index.to_numpy()
    acc = requests["account_id"].to_numpy()
    order = np.lexsort((rng.random(len(idx)), acc))
    acc_sorted = acc[order]
    first = np.concatenate([[True], acc_sorted[1:] != acc_sorted[:-1]])
    rank = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    keep = order[rank < per_account]
    return requests.loc[np.sort(idx[keep]), ["account_id", "requests", "prompt"]]


def assign_families(requests: pd.DataFrame, per_account: int = 60, seed: int = 0, **kw) -> pd.DataFrame:
    """Sampled rows (account_id, requests, family). Shares computed on the sample extrapolate to volume."""
    sample = sample_prompts(requests, per_account, seed)
    fam = structure_families(sample["prompt"].tolist(), seed=seed, **kw)
    return sample.drop(columns="prompt").assign(family=fam)
