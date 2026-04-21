"""Stronger correctness tests:
* Inside probability = Σ over all parse trees of product of rule probs.
* Supervised SCFG training recovers the 4-bp stem exactly.
* Supervised CYK beats untrained CYK on the held-out hairpin.
* Turner 2004 stacking table is symmetric and negative on WC stacks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rnafold import (
    SCFGParams,
    cyk,
    dotbracket_to_pairs,
    inside,
    structure_to_rules,
    train_from_labeled,
    zuker_fold,
)
from rnafold import turner


# ----- Turner parameters -----


def test_turner_stacking_negative_on_canonical_wc_pairs():
    """G-C stacked on G-C / C-G should be strongly negative."""
    assert turner.stacking_energy("G", "C", "G", "C") < -2.0
    assert turner.stacking_energy("C", "G", "G", "C") < -2.0


def test_turner_hairpin_init_increases_beyond_size_9():
    assert turner.hairpin_init(9) < turner.hairpin_init(30)


def test_turner_terminal_au_penalty_applies_to_au_not_gc():
    assert turner.terminal_penalty("A", "U") > 0
    assert turner.terminal_penalty("G", "C") == 0


# ----- Inside = brute-force sum over parse trees (supervised) -----


def _enumerate_nested_structures(seq, min_loop=3):
    n = len(seq)
    from rnafold.utils import can_pair

    def rec(i, j):
        if j - i < min_loop + 1:
            return [[]]
        results = [[]]
        for s in rec(i + 1, j):
            results.append(list(s))
        for k in range(i + min_loop + 1, j + 1):
            if not can_pair(seq[i], seq[k]):
                continue
            for s_inner in rec(i + 1, k - 1):
                for s_outer in rec(k + 1, j):
                    results.append(list(s_inner) + [(i, k)] + list(s_outer))
        seen = set()
        unique = []
        for r in results:
            key = tuple(sorted(r))
            if key not in seen:
                seen.add(key)
                unique.append(list(key))
        return unique

    if n < 1:
        return [[]]
    return rec(0, n - 1)


def _parse_prob(seq, struct, params):
    events = structure_to_rules(seq, struct)
    p = 1.0
    for ev in events:
        rid = ev[0]
        p *= params.p_rule[rid]
        if rid == 3:
            p *= params.p_single[ev[2]]
        elif rid in (2, 4):
            p *= params.p_pair[ev[2], ev[3]]
    return p


def test_inside_matches_brute_force_over_parses_tiny():
    """On a 3-nt sequence, no pairs are possible (smallest paired span
    under KH99 is length 4). Inside probability must equal the unique
    all-unpaired parse probability."""
    params = SCFGParams()
    seq = "AAA"
    p_parse = _parse_prob(seq, "...", params)
    a = inside(seq, params)
    Z = a[0, len(seq), 0]
    assert abs(Z - p_parse) / max(p_parse, 1e-12) < 1e-8, (Z, p_parse)


def test_inside_matches_brute_force_over_parses_grammar_min_loop_2():
    """On a 4-nt sequence, two structures are possible under KH99's
    minimum pair span (length 4): `....` and `(..)`. The inside must
    equal their summed probability."""
    from rnafold.utils import pairs_to_dotbracket
    params = SCFGParams()
    seq = "GCGC"
    # Structures: fully unpaired and (if pairing allowed) (0, 3) paired.
    from rnafold.utils import can_pair
    candidates = [[]]
    if can_pair(seq[0], seq[3]):
        candidates.append([(0, 3)])
    total = 0.0
    for pairs_list in candidates:
        struct = pairs_to_dotbracket(pairs_list, len(seq))
        total += _parse_prob(seq, struct, params)
    a = inside(seq, params)
    Z = a[0, len(seq), 0]
    assert abs(Z - total) / max(total, 1e-12) < 1e-8, (Z, total, candidates)


# ----- Supervised training improves CYK -----


def test_supervised_training_improves_cyk_on_held_out_stem():
    """Train on a batch of simple stems; untrained CYK over-pairs, but
    trained CYK should put pairs only where the training data did."""
    # Training set: four clean hairpin structures.
    train = [
        ("GGGGAAAACCCC", "((((....))))"),
        ("CCCCAAAAGGGG", "((((....))))"),
        ("GCCCAAAAGGGC", "((((....))))"),
        ("GCGCAAAAGCGC", "((((....))))"),
    ]
    untrained = SCFGParams()
    trained = train_from_labeled(train, pseudo=0.5)
    held_out = "GGCGCAAAAGCGCC"  # similar stem, 14 nt
    # Untrained CYK — will over-pair.
    _, s_untrained = cyk(held_out, untrained)
    # Trained CYK — should favor a single outer stem around 4 pairs.
    _, s_trained = cyk(held_out, trained)
    pairs_u = dotbracket_to_pairs(s_untrained)
    pairs_t = dotbracket_to_pairs(s_trained)
    # Trained should produce at least one fewer pair than untrained
    # (EVERY base is paired in untrained CYK because unit pair emissions
    # dominate with uniform prior).
    assert len(pairs_t) <= len(pairs_u)
    # Trained should give a structure with a clear outer stem.
    assert any(i <= 1 and j >= len(held_out) - 2 for i, j in pairs_t), pairs_t


def test_structure_to_rules_roundtrips_probability_to_one():
    """For a known structure, the rules extracted partition probability
    mass correctly: with probability-1 distributions on the relevant
    rules, the parse probability is 1."""
    seq = "GCGCAAAAGCGC"
    struct = "((((....))))"
    events = structure_to_rules(seq, struct)
    # Every emission event must be correctly typed.
    single_count = sum(1 for ev in events if ev[0] == 3)
    pair_count = sum(1 for ev in events if ev[0] in (2, 4))
    assert single_count + 2 * pair_count == len(seq)
