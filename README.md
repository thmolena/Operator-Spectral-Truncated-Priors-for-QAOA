# Operator-Spectral Truncated Priors for Query-Efficient QAOA (OST-QAOA)

[![Project Website](https://img.shields.io/badge/Project_Website-Open-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)
[![Paper](https://img.shields.io/badge/Paper-OST--QAOA%20(PDF)-1d4ed8?style=for-the-badge)](submission/main.pdf)
[![Code Artifact](https://img.shields.io/badge/Reproducible_Code-submission%2Fcode-7c3aed?style=for-the-badge)](submission/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-111827?style=for-the-badge)](LICENSE)

This repository develops **OST-QAOA**, a *noncommutative operator-spectral truncated prior* for query-efficient parameter selection in the Quantum Approximate Optimization Algorithm (QAOA). It transports the spectral-truncation–kernel construction of noncommutative $C^{*}$-algebraic kernel machines into variational quantum optimization, and is released as the installable package [`uq-qaoa`](submission/code) that regenerates every figure, table, and number in the [manuscript](submission/main.pdf) from a single deterministic seed.

---

## Principal contribution

> **Manuscript:** *Operator-Spectral Truncated Priors for Query-Efficient QAOA Parameter Search* — [`submission/main.pdf`](submission/main.pdf).
> **Reproducible artifact:** [`submission/code/`](submission/code).

For low-depth QAOA the dominant practical cost is the number of objective-function *queries* required to tune the variational angles on each graph. A warm-start point alone is insufficient: the optimizer must also decide *which angle directions* deserve exploration under a fixed query budget. OST-QAOA answers this by learning and searching in **operator space** rather than in parameter-vector space.

### Method

For a graph $G=(V,E)$ the MaxCut cost Hamiltonian and depth-$p$ state are

$$
H_C(G)=\sum_{(i,j)\in E}\tfrac{1-Z_iZ_j}{2},
\qquad
\lvert\psi_G(\theta)\rangle=\prod_{\ell=1}^{p}e^{-i\beta_\ell H_M}\,e^{-i\gamma_\ell H_C(G)}\,\lvert+\rangle^{\otimes n},
$$

with objective $f_G(\theta)=\langle\psi_G(\theta)\lvert H_C(G)\rvert\psi_G(\theta)\rangle$ and approximation ratio $r_G=f_G/C_G^\star$. OST-QAOA maps $G$ to two **noncommuting** $2p\times2p$ angle-space operators $\mathcal A_G,\mathcal B_G$ built from the graph Laplacian spectrum, degree moments, and topology features. Their commutator $\mathcal C_G=[\mathcal A_G,\mathcal B_G]$ enters a single raw operator $\mathcal O_G$, which is **spectrally truncated** to a rank-$n$ covariance

$$
\mathcal O_{G,r}=U_r\Lambda_r U_r^{\top},
\qquad
\Sigma_G=\beta\,\frac{\mathcal O_{G,r}}{\mathrm{tr}\,\mathcal O_{G,r}}+\Bigl(1-\tfrac{\beta}{2}\Bigr)\frac{\Sigma_G^{\mathrm{nbr}}}{\mathrm{tr}\,\Sigma_G^{\mathrm{nbr}}}+\epsilon I .
$$

The truncated, operator-dominated covariance ($\beta=0.8$) supplies the **collective, cross-coordinate search directions**: the query policy probes the leading eigenvectors of $\Sigma_G$ — not the raw angle coordinates — around the operator-derived posterior mean, with step sizes proportional to the directional standard deviation.

### Theory

The manuscript proves that $\Sigma_G$ is positive definite and Loewner-contractive, that the construction reduces to the **commutative diagonal prior** in the single-direction limit ($n\to1$), and that the expected number of objective queries to reach a target ratio scales with the prior's **effective dimension** $d_{\mathrm{eff}}(n)=(\sum_j\lambda_j)^2/\sum_j\lambda_j^2$, which the truncation minimizes. This is the *query-budget analogue* of the representation-versus-complexity tradeoff of spectral-truncation kernels: the truncation parameter $n$ trades near-optimal coverage against the number of objective queries, and the predicted **interior optimum** $n^\star$ is observed empirically.

### Main result (exact-statevector MaxCut, $p{=}3$, $Q{=}24$, rank $4$, commutator weight $4.0$, seed $260424803$, 16 held-out graphs)

Mean approximation ratio with 95% interval and paired difference versus a budget-matched TQA coordinate-refinement baseline, transcribed from [`submission/code/tables/table01_headline.csv`](submission/code/tables/table01_headline.csv):

| Method | Mean ratio ↑ | $\Delta$ vs. TQA+coord. [95% CI] | Wins | $\bar Q_{0.98}$ ↓ |
|---|---|---|---|---|
| Random | 0.743 ± 0.028 | +0.035 [+0.001, +0.070] | 11/16 | 24.1 |
| TQA | 0.614 ± 0.030 | −0.093 [−0.156, −0.040] | 0/16 | 25.0 |
| TQA + coordinate | 0.708 ± 0.038 | 0.000 | — | 25.0 |
| $k$-NN + coordinate | 0.740 ± 0.042 | +0.032 [+0.012, +0.056] | 13/16 | 23.6 |
| OST diagonal | 0.806 ± 0.020 | +0.099 [+0.067, +0.130] | 15/16 | 14.4 |
| **OST-QAOA (ours)** | **0.818 ± 0.018** | **+0.110 [+0.080, +0.141]** | **16/16** | **11.6** |

OST-QAOA is the strongest method under the matched query budget, winning on **16/16** paired held-out graphs (sign-test $p<10^{-4}$) and reaching 98% of the best observed ratio in **11.6** queries against **25.0** for the baseline — a **2.2× query reduction** ($\bar Q_{0.98}$). A truncation sweep over $n\in\{1,\dots,2p\}$ exhibits the predicted interior optimum at $n^\star{=}4$ (under-truncation at $n{=}1$ and no truncation at $n{=}2p$ both degrade), with effective dimension rising monotonically from 2.50 to 3.65 ([`tables/table05_truncation.csv`](submission/code/tables/table05_truncation.csv)). Restricting the search to diagonal (commutative) directions removes a measurable part of the advantage (0.806, −0.012), localizing the gain to the off-diagonal operator geometry ([`tables/table02_ablation.csv`](submission/code/tables/table02_ablation.csv)). The reported margin is configuration-dependent and the audit-sized exact-statevector benchmark is not a hardware-scale deployment claim; it is a controlled, fully reproducible method study.

---

## Reproduce the manuscript end-to-end

```bash
cd submission/code
pip install .                                   # installs the uqqaoa-reproduce / uqqaoa-artifacts entry points

# Regenerate every manuscript figure, table, CSV summary, and query trace:
uqqaoa-reproduce --output-dir . --depth 3 --budget 24 --rank 4 --commutator-weight 4.0

uqqaoa-reproduce --smoke                         # fast end-to-end sanity pass
```

Every output is a deterministic function of the global seed `260424803`. The same functionality is available through the Python API:

```python
from uq_qaoa import build_operator_library, ost_qaoa_search
from uq_qaoa.graphs import generate_graph

library = build_operator_library(p=3, rank=4, commutator_weight=4.0, train_per_family=6)
graph   = generate_graph("random_regular", 10, 1234)
result  = ost_qaoa_search(graph, library, budget=24)   # result.theta_hat, result.y_hat, result.trace
```

The mapping from each manuscript display item to its generating function is listed in the manuscript's artifact manifest and in [`submission/code/README.md`](submission/code/README.md).

---

## Repository layout

```text
submission/
  main.tex         OST-QAOA manuscript (operator construction, theory, experiments)
  main.pdf         compiled manuscript
  code/            installable package `uq-qaoa` (python/uq_qaoa/), figure/table/CSV
                   artifacts, configs, results, and the deterministic artifact driver
notebooks/         companion graph-conditioned learning analyses (broader work)
experiments/ src/  baseline scripts, models, simulators, and serving code (broader work)
data/ outputs/     source inputs and processed results (broader work)
website/ index.html project landing page (GitHub Pages entry point)
```

---

## Citation

```bibtex
@misc{huynh2026ostqaoa,
  author       = {Huynh, Molena},
  title        = {Operator-Spectral Truncated Priors for Query-Efficient QAOA},
  year         = {2026},
  howpublished = {Code and manuscript},
  note         = {North Carolina State University; molena.huynh@jmp.com},
  url          = {https://github.com/thmolena/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization}
}
```

## License

Released under the [MIT License](LICENSE).
