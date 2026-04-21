# Project Plan — RNA Secondary Structure & SCFGs

## Goal
Build a from-scratch library (`rnafold`) that implements the full stack
of classical RNA secondary-structure prediction algorithms: Nussinov
base-pair maximization → Zuker MFE → McCaskill partition function →
full Knudsen-Hein SCFG with Inside/Outside and EM training. Validate
against real tRNA sequences with known structures and compare with
ViennaRNA.

## Layers

### Layer 0 — Foundations (combinatorics + DP)
* RNA alphabet, pairing rules (Watson-Crick + wobble)
* Dot-bracket ↔ pair-list conversions
* Sequence / structure I/O (FASTA + known-structure parsing)

### Layer 1 — Nussinov (O(n³))
* `nussinov_dp(seq)` — fill matrix with 4 cases (unpaired-i,
  unpaired-j, pair, bifurcation)
* `nussinov_traceback(dp, seq)` — recover structure
* Validated against brute-force enumeration on n ≤ 8

### Layer 2 — Zuker MFE with Turner-style parameters
* V(i, j) = MFE assuming (i,j) paired
* W(i, j) = MFE with no restriction (outer region)
* WM(i, j) = MFE of a multi-branch loop piece
* Decompose V(i,j) into hairpin / stack / bulge / internal / multi
* Traceback yields MFE structure
* Validate: MFE ≤ Nussinov pair-count (weighted), equals ViennaRNA's
  RNAfold on small test sequences

### Layer 3 — McCaskill partition function + inside-outside
* Inside Z(i, j): sum over structures on x_i...x_j
* Outside Z'(i, j): partition function of "everything except x_i...x_j"
* Base-pair probabilities P(i, j) from inside × outside / Z_total
* Visualize as dot plot
* Validate: Σ_structures e^{-E/kT} matches brute-force enumeration on
  n ≤ 10

### Layer 4 — SCFG (KH99 grammar)
Grammar:
  S → L S | L
  L → s F s | s
  F → s F s | L S
with pair-emission and single-emission probabilities.

Algorithms:
* **Inside** α[i, j, N] = P(N ⇒* x_{i:j+1})
* **Outside** β[i, j, N] = P(S ⇒* x_{1:i} N x_{j+1:n+1})
* **CYK** Viterbi analog: argmax over derivations
* **Inside-Outside EM** for parameter re-estimation from data
* Structure extraction from CYK parse tree

Validate:
* Inside matches brute-force summation on n ≤ 7
* CYK-optimal parse of a sequence of known structure recovers it
* EM on a batch of tRNA sequences converges (monotone log-likelihood)

### Layer 5 — Visualization
* Dot-bracket pretty print with alignment markers
* Arc diagram (bases on a line, arcs for pairs)
* Dot plot (upper-triangle heatmap of P(i,j))
* DP matrix heatmap (from Nussinov/Zuker/McCaskill)

### Layer 6 — External validation
* `external/vienna_bridge.py` — optional `ViennaRNA` wrapper
* Golden tests: our MFE within ε of ViennaRNA's on small sequences

## Milestones

| # | Deliverable | Layer | Budget |
|---|-------------|-------|--------|
| M1 | Nussinov + traceback + tests | 0–1 | 1 session |
| M2 | Zuker V/W + proper loops + MFE traceback | 2 | 2 sessions |
| M3 | McCaskill inside-outside + bp probabilities + dot plot | 3 | 1 session |
| M4 | SCFG with CYK + Inside + Outside | 4a | 2 sessions |
| M5 | SCFG parameter training via EM | 4b | 1 session |
| M6 | Visualization module | 5 | 0.5 sessions |
| M7 | Real-data tRNA benchmark + ViennaRNA comparison | 6 | 1 session |

## Data
* `data/rna_known_structures.fasta` — yeast tRNA-Phe (76 nt), CUUCGG
  tetraloop hairpin, simple 4-bp stem, two-hairpin toy sequence — all
  with published dot-bracket structures for validation.

## Stack
* Python 3.10+, NumPy
* matplotlib for visualization
* pytest for tests
* optional: `viennarna` Python bindings (external comparison only)

## Success criteria
* Nussinov matches brute-force enumeration up to n=8 on 100 random seqs
* Zuker MFE recovers the 4-bp stem exactly on `GCGCAAAAGCGC`
* McCaskill Z matches brute-force Z on n ≤ 10
* SCFG Inside matches brute-force summation to 1e-10
* CYK on `GCGCAAAAGCGC` with a reasonable grammar recovers `((((....))))`
* EM log-likelihood monotonically increases
* Figures generated: DP heatmap, arc diagram, dot plot
