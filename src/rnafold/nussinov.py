from __future__ import annotations

import numpy as np

from .utils import can_pair


def nussinov_dp(sequence: str, min_loop: int = 3) -> np.ndarray:
    """Nussinov base-pair maximization DP.

    Enforces a minimum hairpin loop size of ``min_loop`` (no pair (i, j)
    unless j - i > min_loop).

    Complexity: O(n^3) time, O(n^2) space.
    """
    n = len(sequence)
    dp = np.zeros((n, n), dtype=int)

    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            down = dp[i + 1, j] if i + 1 <= j else 0
            left = dp[i, j - 1] if i <= j - 1 else 0
            diag = 0
            if j - i > min_loop and can_pair(sequence[i], sequence[j]):
                inner = dp[i + 1, j - 1] if i + 1 <= j - 1 else 0
                diag = inner + 1
            bifurcation = 0
            for k in range(i, j):
                cand = dp[i, k] + dp[k + 1, j]
                if cand > bifurcation:
                    bifurcation = cand
            dp[i, j] = max(down, left, diag, bifurcation)
    return dp


def nussinov_traceback(dp: np.ndarray, sequence: str, min_loop: int = 3) -> str:
    n = len(sequence)
    pairs = []

    def tb(i: int, j: int) -> None:
        if i >= j:
            return
        if dp[i, j] == dp[i + 1, j]:
            tb(i + 1, j)
            return
        if dp[i, j] == dp[i, j - 1]:
            tb(i, j - 1)
            return
        if j - i > min_loop and can_pair(sequence[i], sequence[j]):
            inner = dp[i + 1, j - 1] if i + 1 <= j - 1 else 0
            if dp[i, j] == inner + 1:
                pairs.append((i, j))
                tb(i + 1, j - 1)
                return
        for k in range(i + 1, j):
            if dp[i, j] == dp[i, k] + dp[k + 1, j]:
                tb(i, k)
                tb(k + 1, j)
                return

    tb(0, n - 1)
    structure = ["."] * n
    for i, j in pairs:
        structure[i], structure[j] = "(", ")"
    return "".join(structure)
