from __future__ import annotations

from typing import Tuple

import numpy as np

from .utils import can_pair


def _bp_energy_toy(b1: str, b2: str) -> float:
    pair = (b1.upper(), b2.upper())
    if pair in (("G", "C"), ("C", "G")):
        return -3.0
    if pair in (("A", "U"), ("U", "A")):
        return -2.0
    if pair in (("G", "U"), ("U", "G")):
        return -1.0
    return float("inf")


def inside_Z(sequence: str, RT: float = 1.0, min_loop: int = 3) -> np.ndarray:
    """Inside partition function Z(i, j) over all nested structures of
    x_i...x_j, under a toy base-pair-only energy model.

    Recursion:
      Z(i, j) = 1                       if j < i or j - i < min_loop
      Z(i, j) = Z(i, j-1)               (j unpaired)
              + Sigma_{i <= k <= j - min_loop - 1} Z(i, k-1) * Z(k+1, j-1) * exp(-E(x_k, x_j)/RT)
                                          (j paired with some k)

    Standard McCaskill recurrence -- the k=j case corresponds to "i
    unpaired, j unpaired" via the Z(i, j-1) term.
    """
    n = len(sequence)
    Z = np.zeros((n + 1, n + 1), dtype=np.float64)
    # Empty subsequences contribute 1 (one empty structure).
    for i in range(n + 1):
        for j in range(i - 1, min(i + min_loop, n) + 1):
            if i <= j + 1:
                Z[i, j] = 1.0
        # Also handle Z[i, i-1] = 1 (empty) and Z[i, i..i+min_loop] = 1
    for length in range(min_loop + 2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            # case: j unpaired
            total = Z[i, j - 1]
            # case: j paired with some k in [i, j - min_loop - 1]
            for k in range(i, j - min_loop):
                if not can_pair(sequence[k], sequence[j]):
                    continue
                E = _bp_energy_toy(sequence[k], sequence[j])
                w = np.exp(-E / RT)
                left = Z[i, k - 1] if k >= 1 else 1.0
                inside = Z[k + 1, j - 1]
                total += left * inside * w
            Z[i, j] = total
    return Z


def outside_Z(sequence: str, Z_in: np.ndarray, RT: float = 1.0, min_loop: int = 3) -> np.ndarray:
    """Outside partition function Z'(i, j): partition function over all
    structures that enclose x_i...x_j as an inner region (or "the region
    outside of [i, j]"). Used together with Z_in to compute P(i, j).
    """
    n = len(sequence)
    Z_out = np.zeros((n + 1, n + 1), dtype=np.float64)
    total_Z = Z_in[0, n - 1]
    # Seed: outside of the full range is 1 (the empty context).
    Z_out[0, n - 1] = 1.0

    # Fill decreasing in length -- the outside recursion relies on larger
    # "surrounding" regions having already been filled.
    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            # (1) j+1 unpaired: a larger structure [i, j+1] has j+1
            # unpaired, contributing the outside of [i, j+1] * 1 to Z_out[i,j].
            if j + 1 < n:
                Z_out[i, j] += Z_out[i, j + 1]  # x_{j+1} unpaired
            # (2) i-1 unpaired -- there's no such direct contribution in the
            # standard outside recursion for this grammar; the symmetric
            # "i-1 unpaired" case is handled through (3).
            # (3) A base pair (k, j+1) for some k <= i ? 1, enclosing [i, j]:
            if j + 1 < n:
                for k in range(0, i):
                    if not can_pair(sequence[k], sequence[j + 1]):
                        continue
                    E = _bp_energy_toy(sequence[k], sequence[j + 1])
                    w = np.exp(-E / RT)
                    # The outside of [k, j+1] times the inside of the
                    # partial left region Z[k+1, i-1] contributes.
                    left = Z_in[k + 1, i - 1] if k + 1 <= i - 1 else 1.0
                    Z_out[i, j] += Z_out[k, j + 1] * left * w
    return Z_out


def bp_probabilities(sequence: str, RT: float = 1.0, min_loop: int = 3) -> np.ndarray:
    """Return P(i, j) = probability that (i, j) is paired in the Boltzmann
    ensemble, using the toy base-pair-only energy model.

    P(i, j) = [Z_out(i, j) * Z_in(i+1, j-1) * exp(-E/RT)] / Z_total
    """
    n = len(sequence)
    Z = inside_Z(sequence, RT=RT, min_loop=min_loop)
    Zo = outside_Z(sequence, Z, RT=RT, min_loop=min_loop)
    total = Z[0, n - 1]
    P = np.zeros((n, n), dtype=np.float64)
    if total <= 0:
        return P
    for i in range(n):
        for j in range(i + min_loop + 1, n):
            if not can_pair(sequence[i], sequence[j]):
                continue
            E = _bp_energy_toy(sequence[i], sequence[j])
            w = np.exp(-E / RT)
            inside = Z[i + 1, j - 1] if i + 1 <= j - 1 else 1.0
            P[i, j] = Zo[i, j] * inside * w / total
            P[j, i] = P[i, j]
    return P


def mccaskill_partition(sequence: str, RT: float = 1.0, min_loop: int = 3) -> float:
    """Total partition function -- convenience wrapper returning Z[0, n-1]."""
    Z = inside_Z(sequence, RT=RT, min_loop=min_loop)
    return float(Z[0, len(sequence) - 1])
