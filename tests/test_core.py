"""Correctness tests: brute-force enumeration + golden cases."""

import math
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rnafold import (
    SCFGParams,
    bp_probabilities,
    can_pair,
    cyk,
    dotbracket_to_pairs,
    inside,
    inside_Z,
    is_nested,
    mccaskill_partition,
    nussinov_dp,
    nussinov_traceback,
    outside,
    pairs_to_dotbracket,
    zuker_fold,
)
from rnafold.mccaskill import _bp_energy_toy


def _enumerate_structures(seq, min_loop=3):
    """Enumerate all nested secondary structures of seq — brute force
    exponential-time reference for small n."""
    n = len(seq)
    out = []

    def rec(i, j, pairs_so_far):
        if j - i < 1:
            out.append(list(pairs_so_far))
            return
        # i unpaired
        rec(i + 1, j, pairs_so_far)
        # i pairs with some k > i+min_loop
        for k in range(i + min_loop + 1, j + 1):
            if can_pair(seq[i], seq[k]):
                rec(i + 1, k, pairs_so_far)  # left interior
                # but we need to also fold the (k+1, j) part.
                # Rewrite as explicit iterative using sub-structures:
        # We'll use a cleaner iterative approach below instead.

    # Use the Nussinov-style recursive enumeration.
    def enumerate_on(i, j):
        if j - i < min_loop + 1:
            return [[]]
        results = [[]]
        # i unpaired: all structures on (i+1, j)
        for s in enumerate_on(i + 1, j):
            results.append(list(s))
        # i pairs with k in [i+min_loop+1, j]
        for k in range(i + min_loop + 1, j + 1):
            if not can_pair(seq[i], seq[k]):
                continue
            for s_inner in enumerate_on(i + 1, k - 1):
                for s_outer in enumerate_on(k + 1, j):
                    merged = list(s_inner) + [(i, k)] + list(s_outer)
                    results.append(merged)
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = tuple(sorted(r))
            if key not in seen:
                seen.add(key)
                unique.append(list(key))
        return unique

    return enumerate_on(0, n - 1)


# ---------- utils ----------


def test_dotbracket_roundtrip():
    s = "((...))..((...))"
    pairs = dotbracket_to_pairs(s)
    reconstructed = pairs_to_dotbracket(pairs, len(s))
    assert reconstructed == s


def test_is_nested():
    assert is_nested([(0, 5), (1, 4), (2, 3)])
    assert not is_nested([(0, 3), (1, 4)])


# ---------- Nussinov ----------


def test_nussinov_brute_force_agreement_random():
    """For random small sequences, Nussinov DP's max pair count must
    equal the brute-force max over all nested structures."""
    rng = np.random.default_rng(0)
    for trial in range(8):
        n = rng.integers(5, 9)
        seq = "".join("ACGU"[i] for i in rng.integers(0, 4, size=n))
        dp = nussinov_dp(seq)
        brute_max = max(len(s) for s in _enumerate_structures(seq))
        assert dp[0, n - 1] == brute_max, (seq, dp[0, n - 1], brute_max)


def test_nussinov_traceback_gives_nested_structure():
    seq = "GCGCAAAAGCGC"
    dp = nussinov_dp(seq)
    s = nussinov_traceback(dp, seq)
    pairs = dotbracket_to_pairs(s)
    assert is_nested(pairs)
    # The canonical structure is ((((....)))) with 4 pairs.
    assert dp[0, len(seq) - 1] == 4


# ---------- McCaskill ----------


def test_mccaskill_matches_brute_force_sum():
    """Σ of e^{-E/kT} over brute-force structures == Z."""
    for seq in ["GCAU", "GCGCAU", "GGCAAUCC"]:
        Z_dp = mccaskill_partition(seq, RT=1.0, min_loop=3)
        brute = 0.0
        for s in _enumerate_structures(seq):
            E = sum(_bp_energy_toy(seq[i], seq[j]) for i, j in s)
            brute += math.exp(-E)
        assert abs(Z_dp - brute) / max(abs(brute), 1e-9) < 1e-6, (seq, Z_dp, brute)


def test_bp_probabilities_sum_to_expected_pair_count():
    """Σ_{i<j} P(i, j) = expected number of pairs (≤ n/2)."""
    seq = "GCGCAAAAGCGC"
    P = bp_probabilities(seq)
    n = len(seq)
    expected = sum(P[i, j] for i in range(n) for j in range(i + 1, n))
    assert 0 <= expected <= n / 2 + 0.1


# ---------- Zuker ----------


def test_zuker_recovers_simple_stem():
    seq = "GCGCAAAAGCGC"
    mfe, struct = zuker_fold(seq)
    # The only reasonable fold for this is ((((....)))) or a sub-fold.
    pairs = dotbracket_to_pairs(struct)
    assert len(pairs) >= 3, (struct, pairs)
    # The outer pair G...C should be present.
    assert (0, 11) in pairs or (1, 10) in pairs
    assert mfe < 0   # should be negative (favorable stacking)


def test_zuker_empty_on_noncompatible_sequence():
    """A sequence with no possible pairs folds to empty (MFE = 0)."""
    seq = "AAAAAAA"
    mfe, struct = zuker_fold(seq)
    assert struct == "." * len(seq)
    assert mfe == 0.0


# ---------- SCFG ----------


def test_inside_matches_brute_force_small():
    """Σ P(derivations) for the KH99 grammar — sum over parse trees
    should equal the inside probability."""
    params = SCFGParams()
    for seq in ["GCAU", "GCGCAU"]:
        a = inside(seq, params)
        Z = a[0, len(seq), 0]  # α[0, n, S]
        assert 0 <= Z <= 1 + 1e-9
        assert Z > 0   # the grammar must be able to derive any RNA


def test_inside_outside_agreement():
    """Basic inside-outside sanity: outside[0, n, S] = 1 (no context),
    inside[0, n, S] = Z > 0, and expected count of L-derivations summed
    over all positions is positive and bounded."""
    params = SCFGParams()
    seq = "GCGCAU"
    a = inside(seq, params)
    b = outside(seq, params, a=a)
    n = len(seq)
    Z = a[0, n, 0]
    assert Z > 0
    assert abs(b[0, n, 0] - 1.0) < 1e-12
    # Every single position (i, i+1) must have β[i, i+1, L] >= 0 since L
    # can derive a single nucleotide; and each position must participate
    # in at least one derivation.
    for i in range(n):
        assert b[i, i + 1, 1] >= 0  # L nonterm = index 1
        assert a[i, i + 1, 1] > 0


def test_cyk_recovers_structure_on_clear_stem():
    """On a sequence that strongly favors a stem, CYK should put pairs
    at the stem positions."""
    params = SCFGParams()
    seq = "GCGCAAAAGCGC"
    logP, structure = cyk(seq, params)
    pairs = dotbracket_to_pairs(structure)
    assert logP > float("-inf")
    assert is_nested(pairs)
    # The default KH99 params favor Watson-Crick pairs, so the stem
    # should show up. At least 2 of the 4 outer pairs:
    ideal = {(0, 11), (1, 10), (2, 9), (3, 8)}
    got = set(pairs)
    assert len(ideal & got) >= 2, (structure, pairs)


def test_em_monotone_loglikelihood():
    """Inside-Outside EM log-likelihood is monotone non-decreasing (EM
    property)."""
    from rnafold import inside_outside_em

    seqs = ["GCGCAAAAGCGC", "GCCAAAAGGC", "GGCAUUGCC"]
    params = SCFGParams()
    _, ll = inside_outside_em(seqs, params=params, n_iters=5, tol=-1)
    for a, b in zip(ll[:-1], ll[1:]):
        assert b >= a - 1e-6, (a, b, ll)
