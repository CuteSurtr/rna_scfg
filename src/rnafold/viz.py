from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .utils import dotbracket_to_pairs


def plot_matrix(matrix: np.ndarray, title: str = "", ax: Optional[plt.Axes] = None,
                cmap: str = "viridis", show_colorbar: bool = True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    m = np.array(matrix, dtype=float)
    if np.isinf(m).any():
        m = np.where(np.isinf(m), np.nan, m)
    im = ax.imshow(m, cmap=cmap, interpolation="none")
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    return ax


def plot_dot_plot(P: np.ndarray, sequence: str = "", ax: Optional[plt.Axes] = None,
                   title: str = "Base-pair probability dot plot"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(P, cmap="magma", origin="upper", vmin=0, vmax=1)
    ax.set_title(title)
    if sequence:
        n = len(sequence)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(list(sequence), fontsize=7)
        ax.set_yticklabels(list(sequence), fontsize=7)
    return ax


def plot_arc_diagram(sequence: str, structure: str, ax: Optional[plt.Axes] = None,
                     title: str = "Secondary structure arc diagram"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    n = len(sequence)
    pairs = dotbracket_to_pairs(structure) if structure else []
    # baseline bases
    for i, ch in enumerate(sequence):
        ax.text(i, 0, ch, ha="center", va="center", fontsize=8)
    ax.plot(range(n), [0] * n, ".", color="black", markersize=3)
    # arcs
    for i, j in pairs:
        center = (i + j) / 2
        width = j - i
        height = width / 2
        theta = np.linspace(0, np.pi, 50)
        xs = center + (width / 2) * np.cos(np.pi - theta)
        ys = height * np.sin(theta) * 0.5
        ax.plot(xs, ys, color="tab:blue", linewidth=1)
    ax.set_xlim(-1, n)
    ax.set_ylim(-1, (n / 4) if pairs else 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title)
    return ax


def plot_loglik_trace(ll: Sequence[float], ax: Optional[plt.Axes] = None,
                       title: str = "Inside-Outside EM log-likelihood"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ll, color="tab:green", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def print_structure(seq: str, struct: str):
    print(f"sequence:  {seq}")
    print(f"structure: {struct}")
