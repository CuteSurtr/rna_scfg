# Literature Synthesis — Project 3: RNA Secondary Structure & SCFGs

## 1. The problem

Given an RNA primary sequence x = x_1 ... x_n ∈ {A,C,G,U}^n, predict its
secondary structure: the set of **base pairs** S ⊂ {(i,j) : i < j} that
are nested (non-crossing) and biologically feasible (Watson-Crick A-U,
G-C and wobble G-U). The secondary structure determines much of the
molecule's function (tRNA cloverleaf, rRNA folds, miRNA hairpins).

Secondary structure prediction sits at the intersection of:
* **combinatorics** (nested matchings / Motzkin paths / Catalan structures)
* **dynamic programming** (Nussinov, Zuker, McCaskill recursions)
* **statistical mechanics** (partition functions, Boltzmann ensembles)
* **formal language theory** (SCFGs, inside-outside training)

## 2. Nussinov (1978, 1980) — maximum-base-pair matching

**Nussinov & Jacobson, PNAS 1980.** The first efficient algorithm
(O(n³) time, O(n²) space). Define

  γ(i, j) = max |S| over nested matchings on x_i ... x_j.

Recurrence (Nussinov):
  γ(i, j) = max{ γ(i+1, j),                        # i unpaired
                γ(i, j-1),                          # j unpaired
                γ(i+1, j-1) + δ(x_i, x_j),          # i,j paired
                max_{i<k<j} γ(i, k) + γ(k+1, j) }   # bifurcation

with δ = 1 if x_i–x_j is a valid pair, else 0.

Traceback gives the structure. **Limitations:** treats all base pairs
equally — biologically naive. Doesn't model loop penalties or helix
stacking bonuses.

## 3. Zuker (1981) — Minimum Free Energy (MFE) folding

Moves from pair counts to thermodynamic **free energy** ΔG. The
Turner energy model assigns free-energy contributions to:
* **stacking pairs** (most favorable, −1 to −3 kcal/mol)
* **hairpin loops** (unfavorable, +3 to +8)
* **bulge loops, internal loops** (unfavorable)
* **multi-branch loops** (unfavorable, affine penalty)

Zuker's recursion uses **two** interleaved matrices:
* V(i, j) = MFE of subsequence x_i...x_j given that (i, j) IS paired
* W(i, j) = MFE of subsequence x_i...x_j with no restriction

The V recursion decomposes the closed loop enclosed by (i,j) into
hairpin / stack / bulge / internal / multi-branch cases. Modern
implementations (ViennaRNA, RNAfold, mfold) add WM (multi-branch
helper) matrices.

Complexity: O(n³) in time (O(n⁴) if you allow arbitrary internal
loop sizes; usually capped at ≤ 30 nt).

## 4. McCaskill (1990) — partition function + base-pair probabilities

Replaces Zuker's min with sum-of-exponentials:
  Z(i, j) = Σ_structures exp(−ΔG / kT)

Two DP passes:
* **Inside (forward) pass:** Z(i, j) = partition function of x_i...x_j
* **Outside (backward) pass:** complement conditioning on being
  enclosed by a pair (i, j).

Yields **base-pair probabilities**:
  P(i, j) = Z_inside(i+1, j-1) · exp(−ΔG_pair/kT) · Z_outside(i, j) / Z_total

Visualized as a **dot plot**. Essential because the MFE structure
isn't always biologically relevant — the Boltzmann ensemble captures
fold diversity and stability.

## 5. Stochastic Context-Free Grammars — Sakakibara 1994, Eddy-Durbin 1994

**The problem HMMs can't solve.** An HMM generates sequences
left-to-right; it can't represent long-range correlations like
"position i pairs with position j". **Context-free grammars** can.

### Sakakibara et al. 1994 — first tRNA CM
Proposed using a probabilistic CFG to capture tRNA consensus, with
Baum-Welch-like EM training (inside-outside algorithm).

### Eddy & Durbin 1994 — covariance models (CMs)
**Durbin book chapter 10** develops this formally. A CM is an SCFG
whose productions are derived from a multiple RNA alignment's
consensus structure. Each paired column (i, j) becomes a "pair
emission" production, each unpaired column a "single emission"
production.

### The canonical algorithms

Let G = (N, Σ, R, S) be a (P)SCFG in Chomsky-like form. The three
fundamental problems (analogous to HMMs):

| Problem | HMM | SCFG |
|---------|-----|------|
| Likelihood P(x\|θ) | forward | **Inside** (O(n³N)) |
| Most likely parse | Viterbi | **CYK** (O(n³N)) |
| Posterior P(rule\|x) | fwd-bwd | **Inside-Outside** (O(n³N²)) |
| Parameter re-estimation | Baum-Welch | **EM via inside-outside** |

### Knudsen-Hein 1999 / Pfold 2003

Proposed the **KH99 grammar** — a small, unambiguous SCFG:

  S → L S | L
  L → s F s | s            (s = single nucleotide)
  F → s F s | L S

Paired emissions from `s F s` productions emit a base pair (x_i, x_j).
The single-emission `s` productions emit unpaired nucleotides.
Unambiguous means each secondary structure has a unique parse, so
posterior probabilities correspond directly to structure
probabilities.

### Dowell-Eddy 2004 — lightweight SCFG benchmarks
Compared 9 different lightweight grammars. Key finding: grammar
choice matters less than thermodynamic parameters, but unambiguous
grammars work best for statistical reasoning.

## 6. Modern landscape

* **ViennaRNA** (Lorenz et al. 2011) — industry-standard Zuker+McCaskill
  implementation with Turner 2004 parameters.
* **INFERNAL / Rfam** (Eddy lab) — SCFG-based covariance model search
  over the Rfam database of ~4000 RNA families.
* **Neural methods** (SPOT-RNA, UFold, RNAfold-ML) — hybrid deep
  learning + dynamic programming; complementary to classical methods.

## Reading list (files in `literature/`)

| File | Role |
|------|------|
| Eddy_Durbin_1994.pdf | foundational CM / SCFG paper |
| Durbin_biological_sequence_analysis.pdf | Durbin book (ch 10: RNA structure analysis) |
