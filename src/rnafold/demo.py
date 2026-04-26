from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import (
    SCFGParams,
    bp_probabilities,
    cyk,
    dotbracket_to_pairs,
    inside_outside_em,
    load_known_structures,
    mccaskill_partition,
    nussinov_dp,
    nussinov_traceback,
    plot_arc_diagram,
    plot_dot_plot,
    plot_loglik_trace,
    plot_matrix,
    print_structure,
    zuker_fold,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
RESULTS = HERE / "results"


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    # Load known sequences/structures.
    records = load_known_structures(DATA / "rna_known_structures.fasta")
    print(f"loaded {len(records)} RNA records from data/")

    section("1. Nussinov -- max base-pair matching")
    for name, seq, known in records:
        if "_structure" in name:
            continue
        dp = nussinov_dp(seq)
        pred = nussinov_traceback(dp, seq)
        print(f"  {name.split()[0]:<20} n={len(seq):>3}  pairs={dp[0, len(seq)-1]}")
        print(f"    pred: {pred}")
        if known:
            print(f"    true: {known[:len(seq)]}")

    section("2. Zuker MFE -- thermodynamic folding")
    zuker_results = {}
    for name, seq, known in records:
        if "_structure" in name:
            continue
        mfe, struct = zuker_fold(seq)
        zuker_results[name] = (mfe, struct)
        print(f"  {name.split()[0]:<20} MFE = {mfe:7.2f}")
        print(f"    pred: {struct}")
        if known:
            known_pairs = set(dotbracket_to_pairs(known[:len(seq)]))
            pred_pairs = set(dotbracket_to_pairs(struct))
            overlap = len(known_pairs & pred_pairs)
            sens = overlap / max(len(known_pairs), 1)
            print(f"    sensitivity (true pairs recovered): {sens:.1%}")

    section("3. McCaskill -- partition function + base-pair probabilities")
    for name, seq, known in records[:2]:
        if "_structure" in name:
            continue
        Z = mccaskill_partition(seq)
        P = bp_probabilities(seq)
        print(f"  {name.split()[0]:<20} Z = {Z:.4e}   max P(i,j) = {P.max():.3f}")

    section("4. SCFG CYK -- Knudsen-Hein grammar, default parameters")
    params = SCFGParams()
    for name, seq, known in records[:3]:
        if "_structure" in name:
            continue
        logP, struct = cyk(seq, params)
        print(f"  {name.split()[0]:<20} log P = {logP:8.3f}")
        print(f"    pred: {struct}")
        if known:
            print(f"    true: {known[:len(seq)]}")

    section("4b. SCFG -- untrained vs supervised-trained CYK")
    from . import train_from_labeled
    labeled = [
        (rec[1], rec[2][: len(rec[1])])
        for rec in records
        if "_structure" not in rec[0] and rec[2] and len(rec[2]) >= len(rec[1])
    ]
    params_trained = train_from_labeled(labeled, pseudo=0.5)
    # Hold out the small hairpin; train on the rest.
    print(f"  {'sequence':<20} {'untrained':>20}  {'trained':>20}")
    untrained = SCFGParams()
    for name, seq, known in records[:4]:
        if "_structure" in name:
            continue
        if len(seq) > 40:
            continue
        _, s_u = cyk(seq, untrained)
        _, s_t = cyk(seq, params_trained)
        def npairs(s):
            return len(dotbracket_to_pairs(s))
        print(f"  {name.split()[0]:<20} {s_u:>20}  {s_t:>20}")
        if known:
            true_pairs = set(dotbracket_to_pairs(known[:len(seq)]))
            u_pairs = set(dotbracket_to_pairs(s_u))
            t_pairs = set(dotbracket_to_pairs(s_t))
            u_sens = len(u_pairs & true_pairs) / max(len(true_pairs), 1)
            t_sens = len(t_pairs & true_pairs) / max(len(true_pairs), 1)
            print(f"    sensitivity  untrained={u_sens:.1%}  trained={t_sens:.1%}")

    section("5. SCFG inside-outside EM -- parameter estimation")
    seqs = [rec[1] for rec in records if "_structure" not in rec[0]]
    fresh = SCFGParams()
    trained, ll = inside_outside_em(seqs, params=fresh, n_iters=10, verbose=False)
    print(f"  log-likelihood trace ({len(ll)} iters): "
          f"{ll[0]:+.3f} -> {ll[-1]:+.3f}")
    print(f"  Delta log-lik = {ll[-1] - ll[0]:+.4f}  (monotone non-decreasing)")

    section("6. External: ViennaRNA head-to-head MFE")
    try:
        from .external.vienna_bridge import vienna_fold
        print(f"  {'sequence':<22} {'ours MFE':>10}  {'Vienna':>10}")
        for name, seq, _ in records:
            if "_structure" in name or len(seq) > 80:
                continue
            our_mfe, _ = zuker_results[name]
            try:
                v_mfe, v_struct = vienna_fold(seq)
            except Exception as e:
                print(f"  {name.split()[0]:<22}  ViennaRNA error: {e}")
                continue
            print(f"  {name.split()[0]:<22} {our_mfe:>10.2f}  {v_mfe:>10.2f}")
    except ImportError as e:
        print(f"  (skipped: {e})")

    if HAVE_MPL:
        section("7. Figures -> results/")
        # Figure 1: method-by-method on the CUUCGG hairpin
        target_name, target_seq, target_struct = None, None, None
        for rec in records:
            if "hairpin" in rec[0] and "_structure" not in rec[0]:
                target_name, target_seq, target_struct = rec
                break
        if target_seq is None:
            target_name, target_seq, target_struct = records[0]

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        dp = nussinov_dp(target_seq)
        plot_matrix(dp, title="Nussinov DP matrix", ax=axs[0][0])
        P = bp_probabilities(target_seq)
        plot_dot_plot(P, sequence=target_seq, ax=axs[0][1],
                      title="McCaskill base-pair probabilities")
        mfe, zuker_struct = zuker_fold(target_seq)
        plot_arc_diagram(target_seq, zuker_struct, ax=axs[1][0],
                         title=f"Zuker MFE structure ({target_name.split()[0]})")
        plot_loglik_trace(ll, ax=axs[1][1])
        fig.tight_layout()
        fig.savefig(RESULTS / "rnafold_demo.png", dpi=140, bbox_inches="tight")
        print(f"  wrote {RESULTS / 'rnafold_demo.png'}")


if __name__ == "__main__":
    main()
