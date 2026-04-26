"""Stochastic Context-Free Grammars for RNA secondary structure.

Implements the **Knudsen-Hein 1999** (KH99) grammar -- an unambiguous
SCFG whose parse tree corresponds one-to-one with a secondary structure.

Grammar:
    S -> L S | L
    L -> s F s' | s
    F -> s F s' | L S

where:
  * `s` is a single-nucleotide emission from the "unpaired-base"
    distribution p_single(a) over {A, C, G, U}.
  * `s F s'` is a pair emission from the "paired" distribution
    p_pair(a, a') over 4 x 4 possibilities. Only Watson-Crick + wobble
    pairs get non-trivial probability; others are near-zero.

Algorithms implemented:
  * **Inside alpha[i, j, N]** -- total probability that N derives x_{i:j+1}.
  * **Outside beta[i, j, N]** -- total probability that S derives x with
    nonterminal N left in [i:j+1].
  * **CYK gamma[i, j, N]** -- Viterbi analog: max probability over
    derivations; plus traceback to produce the MAP parse tree.
  * **Inside-Outside EM** -- parameter re-estimation.
  * **Structure extraction** -- convert a parse tree to dot-bracket.

We index positions 0-based and use half-open intervals [i, j) with
length j - i >= 1 to match the combinatorics cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ALPHABET = "ACGU"
IDX = {c: i for i, c in enumerate(ALPHABET)}
S_, L_, F_ = 0, 1, 2
NONTERMS = ["S", "L", "F"]


@dataclass
class SCFGParams:
    """Parameters of the KH99 grammar. All probabilities (no log space).

    Rules (indexed in this order for EM bookkeeping):
      0: S -> L S
      1: S -> L
      2: L -> s F s'     (paired; sub-param: pair emission matrix)
      3: L -> s          (unpaired; sub-param: single emission vector)
      4: F -> s F s'     (continuing helix; pair emission)
      5: F -> L S
    """

    p_rule: np.ndarray = field(
        default_factory=lambda: np.array([
            0.5, 0.5,   # S rules
            0.5, 0.5,   # L rules
            0.5, 0.5,   # F rules
        ])
    )
    p_single: np.ndarray = field(default_factory=lambda: np.full(4, 0.25))
    p_pair: np.ndarray = field(
        default_factory=lambda: _default_pair_distribution()
    )

    def __post_init__(self) -> None:
        self.p_rule = np.asarray(self.p_rule, dtype=float)
        self.p_single = np.asarray(self.p_single, dtype=float)
        self.p_pair = np.asarray(self.p_pair, dtype=float)
        self._normalize()

    def _normalize(self) -> None:
        # S rules (0, 1)
        s = self.p_rule[0:2].sum()
        if s > 0:
            self.p_rule[0:2] /= s
        # L rules (2, 3)
        s = self.p_rule[2:4].sum()
        if s > 0:
            self.p_rule[2:4] /= s
        # F rules (4, 5)
        s = self.p_rule[4:6].sum()
        if s > 0:
            self.p_rule[4:6] /= s
        s = self.p_single.sum()
        if s > 0:
            self.p_single /= s
        s = self.p_pair.sum()
        if s > 0:
            self.p_pair /= s


def _default_pair_distribution() -> np.ndarray:
    """Seed pair emission matrix: high probability on Watson-Crick and
    wobble pairs, small nonzero on mismatches so EM can adjust."""
    m = np.full((4, 4), 0.005)  # mismatch prior
    wc = [("A", "U"), ("U", "A"), ("C", "G"), ("G", "C")]
    wo = [("G", "U"), ("U", "G")]
    for a, b in wc:
        m[IDX[a], IDX[b]] = 0.22
    for a, b in wo:
        m[IDX[a], IDX[b]] = 0.05
    m /= m.sum()
    return m


def _encode(seq: str) -> np.ndarray:
    return np.array([IDX[c] for c in seq.upper()], dtype=int)


# ------------------------------------------------------------ Inside ----


def inside(seq: str, params: SCFGParams) -> np.ndarray:
    """Compute alpha[i, j, N] = P(N =>* x_{i:j}) with half-open [i, j).
    Shape (n+1, n+1, 3). alpha[i, j, N] = 0 if j <= i (empty derivation, not
    permitted in this grammar -- no eps rules)."""
    x = _encode(seq)
    n = len(x)
    a = np.zeros((n + 1, n + 1, 3), dtype=float)
    # Length-1: emit a single nucleotide.
    for i in range(n):
        a[i, i + 1, L_] = params.p_rule[3] * params.p_single[x[i]]   # L -> s
        a[i, i + 1, S_] = params.p_rule[1] * a[i, i + 1, L_]         # S -> L
        # F has no single-nucleotide rule, so a[_, _, F_] stays 0 here.

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            # L -> s F s'
            if length >= 3:
                a[i, j, L_] = params.p_rule[2] * params.p_pair[x[i], x[j - 1]] * a[i + 1, j - 1, F_]
            # F -> s F s'
            if length >= 3:
                a[i, j, F_] = params.p_rule[4] * params.p_pair[x[i], x[j - 1]] * a[i + 1, j - 1, F_]
            # F -> L S  (bifurcation)
            if length >= 2:
                a_F_LS = 0.0
                for k in range(i + 1, j):
                    a_F_LS += a[i, k, L_] * a[k, j, S_]
                a[i, j, F_] += params.p_rule[5] * a_F_LS
            # S -> L S
            a_S_LS = 0.0
            for k in range(i + 1, j):
                a_S_LS += a[i, k, L_] * a[k, j, S_]
            a[i, j, S_] = params.p_rule[0] * a_S_LS + params.p_rule[1] * a[i, j, L_]
    return a


# ---------------------------------------------------------- Outside ----


def outside(seq: str, params: SCFGParams, a: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute beta[i, j, N] = probability that S =>* x, passing through N
    at [i, j)."""
    x = _encode(seq)
    n = len(x)
    if a is None:
        a = inside(seq, params)
    b = np.zeros((n + 1, n + 1, 3), dtype=float)
    # Seed: beta[0, n, S] = 1.
    b[0, n, S_] = 1.0

    # Outside recursion -- fill in decreasing length (larger outer regions
    # must be set before smaller inner ones).
    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length

            # Contributions to beta[i, j, S]:
            # (a) via S -> L S with S = outer, L = [i, k], S = [k, j] -- but
            # that's the ROOT rule; reverse-gives beta for L at [i, k] from
            # beta of S at [i, j]. So the contribution to beta[i, j, S] comes
            # from where S appears as the right child of S -> L S:
            #   for k <= i:  beta[i, j, S] += p_rule[0] * beta[k, j, S] * alpha[k, i, L]
            for k in range(0, i):
                b[i, j, S_] += params.p_rule[0] * b[k, j, S_] * a[k, i, L_]
            # S at [i, j] as right child of F -> L S:
            for k in range(0, i):
                b[i, j, S_] += params.p_rule[5] * b[k, j, F_] * a[k, i, L_]

            # beta[i, j, L]:
            # L appears as left child of S -> L S:  for k >= j
            for k in range(j + 1, n + 1):
                b[i, j, L_] += params.p_rule[0] * b[i, k, S_] * a[j, k, S_]
            # L appears alone via S -> L:
            b[i, j, L_] += params.p_rule[1] * b[i, j, S_]
            # L as left child of F -> L S:
            for k in range(j + 1, n + 1):
                b[i, j, L_] += params.p_rule[5] * b[i, k, F_] * a[j, k, S_]

            # beta[i, j, F]:
            # F appears inside L -> s F s' at [i-1, j+1]:
            if i - 1 >= 0 and j + 1 <= n:
                b[i, j, F_] += (
                    params.p_rule[2] * params.p_pair[x[i - 1], x[j]]
                    * b[i - 1, j + 1, L_]
                )
            # F appears inside F -> s F s':
            if i - 1 >= 0 and j + 1 <= n:
                b[i, j, F_] += (
                    params.p_rule[4] * params.p_pair[x[i - 1], x[j]]
                    * b[i - 1, j + 1, F_]
                )
    return b


# ------------------------------------------------------------- CYK ----


def cyk(seq: str, params: SCFGParams) -> Tuple[float, str]:
    """Viterbi-SCFG: find the most probable parse and convert to
    dot-bracket structure. Returns (log P*, dot_bracket_string).

    Implemented in log-space to avoid underflow."""
    x = _encode(seq)
    n = len(x)
    NEG = -np.inf
    # gamma in log space
    g = np.full((n + 1, n + 1, 3), NEG, dtype=float)
    bp = {}  # backpointers: (i, j, N) -> info for traceback

    with np.errstate(divide="ignore"):
        log_rule = np.log(np.where(params.p_rule > 0, params.p_rule, 1e-300))
        log_single = np.log(np.where(params.p_single > 0, params.p_single, 1e-300))
        log_pair = np.log(np.where(params.p_pair > 0, params.p_pair, 1e-300))

    # Length-1
    for i in range(n):
        g[i, i + 1, L_] = log_rule[3] + log_single[x[i]]
        bp[(i, i + 1, L_)] = ("emit_single", i)
        g[i, i + 1, S_] = log_rule[1] + g[i, i + 1, L_]
        bp[(i, i + 1, S_)] = ("rule_S_to_L", i, i + 1)

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length

            # L -> s F s'
            if length >= 3:
                cand = log_rule[2] + log_pair[x[i], x[j - 1]] + g[i + 1, j - 1, F_]
                if cand > g[i, j, L_]:
                    g[i, j, L_] = cand
                    bp[(i, j, L_)] = ("rule_L_pair", i, j)

            # F -> s F s'
            f_score = NEG
            f_choice = None
            if length >= 3:
                cand = log_rule[4] + log_pair[x[i], x[j - 1]] + g[i + 1, j - 1, F_]
                if cand > f_score:
                    f_score = cand
                    f_choice = ("rule_F_pair", i, j)
            # F -> L S
            for k in range(i + 1, j):
                cand = log_rule[5] + g[i, k, L_] + g[k, j, S_]
                if cand > f_score:
                    f_score = cand
                    f_choice = ("rule_F_LS", i, k, j)
            g[i, j, F_] = f_score
            if f_choice is not None:
                bp[(i, j, F_)] = f_choice

            # S -> L S | L
            s_score = NEG
            s_choice = None
            if g[i, j, L_] > NEG:
                cand = log_rule[1] + g[i, j, L_]
                if cand > s_score:
                    s_score = cand
                    s_choice = ("rule_S_to_L", i, j)
            for k in range(i + 1, j):
                cand = log_rule[0] + g[i, k, L_] + g[k, j, S_]
                if cand > s_score:
                    s_score = cand
                    s_choice = ("rule_S_LS", i, k, j)
            g[i, j, S_] = s_score
            if s_choice is not None:
                bp[(i, j, S_)] = s_choice

    # Traceback
    pairs: List[Tuple[int, int]] = []

    def tb(i: int, j: int, N: int) -> None:
        entry = bp.get((i, j, N))
        if entry is None:
            return
        tag = entry[0]
        if tag == "emit_single":
            return
        if tag == "rule_S_to_L":
            tb(entry[1], entry[2], L_)
            return
        if tag == "rule_S_LS":
            _, ii, k, jj = entry
            tb(ii, k, L_)
            tb(k, jj, S_)
            return
        if tag == "rule_L_pair":
            _, ii, jj = entry
            pairs.append((ii, jj - 1))
            tb(ii + 1, jj - 1, F_)
            return
        if tag == "rule_F_pair":
            _, ii, jj = entry
            pairs.append((ii, jj - 1))
            tb(ii + 1, jj - 1, F_)
            return
        if tag == "rule_F_LS":
            _, ii, k, jj = entry
            tb(ii, k, L_)
            tb(k, jj, S_)
            return

    tb(0, n, S_)
    from .utils import pairs_to_dotbracket
    return float(g[0, n, S_]), pairs_to_dotbracket(pairs, n)


# --------------------------------------------------------- EM (IO) ----


def _find_matching(struct: str, i: int) -> int:
    depth = 0
    for k in range(i, len(struct)):
        if struct[k] == "(":
            depth += 1
        elif struct[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"no matching ')' for '(' at position {i}")


def structure_to_rules(seq: str, struct: str):
    """Decompose a (seq, dot-bracket) pair into the unique KH99 parse,
    emitting a sequence of (rule_id, payload) tuples. This is the
    supervised decoder: there is exactly one KH99 parse per valid
    structure because the grammar is unambiguous.

    Rule ids:
      0 S->L S   1 S->L   2 L->s F s'   3 L->s   4 F->s F s'   5 F->L S
    """
    assert len(seq) == len(struct)
    x = _encode(seq)
    events: list = []

    def parse_S(i: int, j: int) -> None:
        if i >= j:
            return
        # Find the extent of the first motif starting at i.
        if struct[i] == ".":
            motif_end = i + 1  # single unpaired base
        elif struct[i] == "(":
            motif_end = _find_matching(struct, i) + 1  # paired block
        else:
            raise ValueError(f"unexpected char {struct[i]!r} at {i}")
        if motif_end == j:
            # S -> L  (single motif covers the whole span)
            events.append((1,))
            parse_L(i, j)
        else:
            # S -> L S
            events.append((0,))
            parse_L(i, motif_end)
            parse_S(motif_end, j)

    def parse_L(i: int, j: int) -> None:
        if j - i == 1:
            # L -> s   (must be unpaired by construction)
            assert struct[i] == ".", f"parse_L on paired at {i}"
            events.append((3, "single", x[i]))
        else:
            # L -> s F s'   (paired block)
            assert struct[i] == "(" and struct[j - 1] == ")", (i, j, struct[i], struct[j - 1])
            events.append((2, "pair", x[i], x[j - 1]))
            parse_F(i + 1, j - 1)

    def parse_F(i: int, j: int) -> None:
        # F covers the interior of a paired block.
        if i >= j:
            # empty interior -- shouldn't happen since min hairpin is 1
            return
        # Does the interior start with a paired block that closes at j-1?
        if struct[i] == "(" and _find_matching(struct, i) == j - 1:
            # F -> s F s'   (helix continues)
            events.append((4, "pair", x[i], x[j - 1]))
            parse_F(i + 1, j - 1)
        else:
            # F -> L S
            events.append((5,))
            # Find the first motif end in [i, j).
            if struct[i] == ".":
                first_motif_end = i + 1
            elif struct[i] == "(":
                first_motif_end = _find_matching(struct, i) + 1
            else:
                # interior can start with ')' only if hairpin loop; but then
                # the structure should have been consumed as F->s F s'. This
                # branch handles the "all unpaired loop" case:
                first_motif_end = j
            parse_L(i, first_motif_end)
            parse_S(first_motif_end, j)

    parse_S(0, len(seq))
    return events


def train_from_labeled(
    pairs: list, pseudo: float = 1.0
) -> SCFGParams:
    """Supervised MLE training from labeled (seq, struct) pairs.

    Each pair is parsed under the KH99 grammar via ``structure_to_rules``
    and the observed rule counts and emission counts are converted into
    probabilities with Laplace pseudocounts.
    """
    rule_counts = np.full(6, pseudo)
    single_counts = np.full(4, pseudo)
    pair_counts = np.full((4, 4), pseudo)

    for seq, struct in pairs:
        if "_structure" in seq:
            continue
        events = structure_to_rules(seq, struct)
        for ev in events:
            rid = ev[0]
            rule_counts[rid] += 1
            if rid == 3:
                single_counts[ev[2]] += 1
            elif rid in (2, 4):
                pair_counts[ev[2], ev[3]] += 1

    return SCFGParams(
        p_rule=rule_counts, p_single=single_counts, p_pair=pair_counts
    )


def inside_outside_em(
    sequences: Sequence[str],
    params: Optional[SCFGParams] = None,
    n_iters: int = 20,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[SCFGParams, List[float]]:
    """Baum-Welch-like training: re-estimate grammar parameters from a
    set of unlabeled RNA sequences using inside-outside expected counts.
    Returns (trained_params, log_likelihood_trace)."""
    if params is None:
        params = SCFGParams()
    ll_trace: List[float] = []
    for it in range(n_iters):
        # Expected rule / emission counts
        rule_counts = np.zeros_like(params.p_rule)
        single_counts = np.zeros_like(params.p_single)
        pair_counts = np.zeros_like(params.p_pair)
        total_ll = 0.0

        for seq in sequences:
            x = _encode(seq)
            n = len(x)
            a = inside(seq, params)
            b = outside(seq, params, a=a)
            Z = a[0, n, S_]
            if Z <= 0:
                continue
            total_ll += np.log(Z)

            # For each rule, accumulate posterior expected count
            # S -> L S: for each i<k<j, alpha[i,k,L]*alpha[k,j,S]*beta[i,j,S]*p_rule[0] / Z
            for length in range(2, n + 1):
                for i in range(0, n - length + 1):
                    j = i + length
                    for k in range(i + 1, j):
                        c = (
                            params.p_rule[0] * a[i, k, L_] * a[k, j, S_] * b[i, j, S_] / Z
                        )
                        rule_counts[0] += c
                        c = (
                            params.p_rule[5] * a[i, k, L_] * a[k, j, S_] * b[i, j, F_] / Z
                        )
                        rule_counts[5] += c
            # S -> L: alpha[i,j,L]*beta[i,j,S]*p_rule[1]/Z
            for length in range(1, n + 1):
                for i in range(0, n - length + 1):
                    j = i + length
                    c = params.p_rule[1] * a[i, j, L_] * b[i, j, S_] / Z
                    rule_counts[1] += c

            # L -> s F s': for length >= 3
            for length in range(3, n + 1):
                for i in range(0, n - length + 1):
                    j = i + length
                    c = (
                        params.p_rule[2] * params.p_pair[x[i], x[j - 1]]
                        * a[i + 1, j - 1, F_] * b[i, j, L_] / Z
                    )
                    rule_counts[2] += c
                    pair_counts[x[i], x[j - 1]] += c
                    c_f = (
                        params.p_rule[4] * params.p_pair[x[i], x[j - 1]]
                        * a[i + 1, j - 1, F_] * b[i, j, F_] / Z
                    )
                    rule_counts[4] += c_f
                    pair_counts[x[i], x[j - 1]] += c_f
            # L -> s (single): length-1 emissions
            for i in range(n):
                c = params.p_rule[3] * params.p_single[x[i]] * b[i, i + 1, L_] / Z
                rule_counts[3] += c
                single_counts[x[i]] += c

        # M-step: normalize counts back into probabilities.
        new_params = SCFGParams(
            p_rule=rule_counts + 1e-12,
            p_single=single_counts + 1e-12,
            p_pair=pair_counts + 1e-12,
        )
        ll_trace.append(total_ll)
        if verbose:
            print(f"iter {it:3d}  log-lik = {total_ll:.4f}")
        if it > 0 and abs(ll_trace[-1] - ll_trace[-2]) < tol:
            params = new_params
            break
        params = new_params
    return params, ll_trace
