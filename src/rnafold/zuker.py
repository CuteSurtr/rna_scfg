from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .turner import (
    MULTI_A as _MULTI_A,
    MULTI_B as _MULTI_B,
    MULTI_C as _MULTI_C,
    hairpin_init,
    internal_loop_energy,
    stacking_energy,
    terminal_penalty,
)
from .utils import can_pair


INF = float("inf")


def _bp_close_energy(b1: str, b2: str) -> float:
    """Contribution when (b1, b2) is the closing pair of a loop: purely
    the terminal AU/GU penalty (the stacking bonus is paid separately by
    the adjacent helix)."""
    if not can_pair(b1, b2):
        return INF
    return terminal_penalty(b1, b2)


def _stack_bonus(outer_b1: str, outer_b2: str, inner_b1: str, inner_b2: str) -> float:
    return stacking_energy(outer_b1, outer_b2, inner_b1, inner_b2)


def _hairpin_energy(size: int, closing_pair_e: float) -> float:
    if size < 3:
        return INF
    return closing_pair_e + hairpin_init(size)


def _internal_loop_energy(u: int, v: int) -> float:
    return internal_loop_energy(u, v)


def zuker_mfe(sequence: str, min_hairpin: int = 3, max_loop: int = 30) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Simplified Zuker MFE with proper V/W/WM matrices and basic
    hairpin/stack/internal/multi-loop handling.

    Returns (mfe, V, W, WM) where V[i,j] = MFE given (i,j) paired,
    W[i,j] = unrestricted MFE of x_i...x_j, and WM[i,j] = MFE of a
    multi-branch-loop segment on x_i...x_j.
    """
    n = len(sequence)
    V = np.full((n, n), INF)
    W = np.full((n, n), INF)
    WM = np.full((n, n), INF)

    for i in range(n):
        W[i, i] = 0.0
        if i + 1 < n:
            W[i, i + 1] = 0.0

    for length in range(min_hairpin + 2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            if can_pair(sequence[i], sequence[j]):
                e_close = _bp_close_energy(sequence[i], sequence[j])
                # hairpin
                hp = _hairpin_energy(j - i - 1, e_close)
                # stack (internal loop with u=v=0)
                stack = INF
                if i + 1 < j - 1 and can_pair(sequence[i + 1], sequence[j - 1]):
                    stack = V[i + 1, j - 1] + _stack_bonus(
                        sequence[i], sequence[j], sequence[i + 1], sequence[j - 1]
                    )
                # internal loop (search for inner pair (p, q))
                internal = INF
                for p in range(i + 1, min(i + 1 + max_loop, j)):
                    for q in range(max(p + min_hairpin + 1, j - max_loop + (p - i - 1)), j):
                        if not can_pair(sequence[p], sequence[q]):
                            continue
                        u = p - i - 1
                        v = j - q - 1
                        if u == 0 and v == 0:
                            continue  # handled by stack
                        if u + v > max_loop:
                            continue
                        internal_e = V[p, q] + e_close + _internal_loop_energy(u, v)
                        if internal_e < internal:
                            internal = internal_e
                # multi-loop: (i, j) encloses at least 2 branches
                multi = INF
                if j - i - 1 >= 2 * (min_hairpin + 2):
                    for k in range(i + 1, j - 1):
                        cand = e_close + _MULTI_A + WM[i + 1, k] + WM[k + 1, j - 1]
                        if cand < multi:
                            multi = cand
                V[i, j] = min(hp, stack, internal, multi)

            # WM recurrence — at least one branch, possibly preceded/followed
            # by unpaired bases and/or more branches.
            best_wm = INF
            # branch = V(p, q) enclosing a helix, charged _MULTI_C
            for p in range(i, j + 1):
                for q in range(p + min_hairpin + 1, j + 1):
                    if V[p, q] == INF:
                        continue
                    # unpaired bases to the left of p cost _MULTI_B each
                    left_cost = (p - i) * _MULTI_B
                    right_cost = (j - q) * _MULTI_B
                    single_branch = V[p, q] + left_cost + right_cost + _MULTI_C
                    if single_branch < best_wm:
                        best_wm = single_branch
            # concat of two WM regions
            for k in range(i, j):
                cand = WM[i, k] + WM[k + 1, j]
                if cand < best_wm:
                    best_wm = cand
            WM[i, j] = best_wm

            # W recurrence: unrestricted MFE of x_i...x_j.
            best_w = 0.0  # empty structure
            # (i, j) paired: W = V(i, j)
            if V[i, j] < best_w:
                best_w = V[i, j]
            # j unpaired
            if i <= j - 1 and W[i, j - 1] < best_w:
                best_w = W[i, j - 1]
            # i unpaired
            if i + 1 <= j and W[i + 1, j] < best_w:
                best_w = W[i + 1, j]
            # bifurcation
            for k in range(i, j):
                cand = W[i, k] + W[k + 1, j]
                if cand < best_w:
                    best_w = cand
            W[i, j] = best_w

    mfe = float(W[0, n - 1]) if n >= 1 else 0.0
    return mfe, V, W, WM


def zuker_traceback(sequence: str, V: np.ndarray, W: np.ndarray, WM: np.ndarray,
                    min_hairpin: int = 3, max_loop: int = 30) -> str:
    """Reconstruct an MFE structure from the V/W matrices."""
    n = len(sequence)
    pairs: List[Tuple[int, int]] = []

    def tb_W(i: int, j: int) -> None:
        if i >= j:
            return
        # empty
        if abs(W[i, j]) < 1e-9:
            return
        # (i, j) paired
        if abs(W[i, j] - V[i, j]) < 1e-9 and can_pair(sequence[i], sequence[j]):
            pairs.append((i, j))
            tb_V(i, j)
            return
        # j unpaired
        if i <= j - 1 and abs(W[i, j] - W[i, j - 1]) < 1e-9:
            tb_W(i, j - 1)
            return
        # i unpaired
        if i + 1 <= j and abs(W[i, j] - W[i + 1, j]) < 1e-9:
            tb_W(i + 1, j)
            return
        # bifurcation
        for k in range(i, j):
            if abs(W[i, j] - (W[i, k] + W[k + 1, j])) < 1e-9:
                tb_W(i, k)
                tb_W(k + 1, j)
                return

    def tb_V(i: int, j: int) -> None:
        if i >= j:
            return
        e_close = _bp_close_energy(sequence[i], sequence[j])
        # stack
        if i + 1 < j - 1 and can_pair(sequence[i + 1], sequence[j - 1]):
            s = V[i + 1, j - 1] + _stack_bonus(
                sequence[i], sequence[j], sequence[i + 1], sequence[j - 1]
            )
            if abs(V[i, j] - s) < 1e-9:
                pairs.append((i + 1, j - 1))
                tb_V(i + 1, j - 1)
                return
        # internal loop
        for p in range(i + 1, min(i + 1 + max_loop, j)):
            for q in range(max(p + min_hairpin + 1, j - max_loop + (p - i - 1)), j):
                if not can_pair(sequence[p], sequence[q]):
                    continue
                u = p - i - 1
                v = j - q - 1
                if u == 0 and v == 0:
                    continue
                cand = V[p, q] + e_close + _internal_loop_energy(u, v)
                if abs(V[i, j] - cand) < 1e-9:
                    pairs.append((p, q))
                    tb_V(p, q)
                    return
        # multi-loop
        if j - i - 1 >= 2 * (min_hairpin + 2):
            for k in range(i + 1, j - 1):
                cand = e_close + _MULTI_A + WM[i + 1, k] + WM[k + 1, j - 1]
                if abs(V[i, j] - cand) < 1e-9:
                    tb_WM(i + 1, k)
                    tb_WM(k + 1, j - 1)
                    return
        # else: hairpin — nothing more to trace.

    def tb_WM(i: int, j: int) -> None:
        if i > j:
            return
        # single branch
        for p in range(i, j + 1):
            for q in range(p + min_hairpin + 1, j + 1):
                if V[p, q] == INF:
                    continue
                left = (p - i) * _MULTI_B
                right = (j - q) * _MULTI_B
                cand = V[p, q] + left + right + _MULTI_C
                if abs(WM[i, j] - cand) < 1e-9:
                    pairs.append((p, q))
                    tb_V(p, q)
                    return
        # bifurcation
        for k in range(i, j):
            cand = WM[i, k] + WM[k + 1, j]
            if abs(WM[i, j] - cand) < 1e-9:
                tb_WM(i, k)
                tb_WM(k + 1, j)
                return

    if n >= 1:
        tb_W(0, n - 1)
    from .utils import pairs_to_dotbracket
    return pairs_to_dotbracket(pairs, n)


def zuker_fold(sequence: str, min_hairpin: int = 3, max_loop: int = 30) -> Tuple[float, str]:
    """High-level: return (mfe_energy, dot_bracket_structure)."""
    mfe, V, W, WM = zuker_mfe(sequence, min_hairpin=min_hairpin, max_loop=max_loop)
    structure = zuker_traceback(sequence, V, W, WM, min_hairpin=min_hairpin, max_loop=max_loop)
    return mfe, structure
