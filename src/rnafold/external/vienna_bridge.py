"""Optional ViennaRNA bridge for reference MFE folding.

`viennarna` ships a Python module named ``RNA``. Import only succeeds
if the system library is installed (``pip install viennarna``).
"""

from __future__ import annotations

from typing import Tuple

try:
    import RNA as _RNA
except ImportError as e:  # pragma: no cover
    raise ImportError("ViennaRNA required; `pip install viennarna`") from e


def vienna_fold(sequence: str) -> Tuple[float, str]:
    """Return (MFE kcal/mol, dot-bracket) from ViennaRNA's RNAfold."""
    fc = _RNA.fold_compound(sequence)
    structure, mfe = fc.mfe()
    return float(mfe), structure


def vienna_bp_probabilities(sequence: str):
    """Return an nxn matrix of base-pair probabilities from ViennaRNA's
    partition function (upper-triangle; symmetric fill)."""
    import numpy as np

    fc = _RNA.fold_compound(sequence)
    fc.pf()
    n = len(sequence)
    P = np.zeros((n, n), dtype=float)
    bpp = fc.bpp()
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            P[i - 1, j - 1] = bpp[i][j]
            P[j - 1, i - 1] = P[i - 1, j - 1]
    return P
