# Hybrid Quantum–Graph AI: Graph-Conditioned Learning for QAOA and Biomedical Optimization

[![Project Website](https://img.shields.io/badge/Project_Website-Open-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)
[![Paper](https://img.shields.io/badge/Paper-UQ--QAOA-1d4ed8?style=for-the-badge)](submission/main.tex)
[![Code Artifact](https://img.shields.io/badge/Reproducible_Code-submission%2Fcode-7c3aed?style=for-the-badge)](submission/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-111827?style=for-the-badge)](LICENSE)

This repository develops graph-conditioned machine learning for variational quantum optimization and for graph-structured biomedical inference. Its unifying thesis is that two apparently distinct decision problems—selecting variational parameters for the Quantum Approximate Optimization Algorithm (QAOA) and predicting node-level clinical risk—admit one formulation: learning a map from a structured graph to a calibrated parameterization of a downstream objective.

The repository is organized around two connected parts together with the artifacts (paper, code, notebooks, website) that support them.

| Part | Scope | Location |
|---|---|---|
| **Query-efficient QAOA parameter search** *(principal contribution)* | Conversion of graph-conditioned predictive uncertainty into an operational trust-region geometry that allocates a fixed QAOA query budget against single-source warm starts and black-box optimizers. | [`submission/`](submission) — manuscript and reproducible code artifact |
| **Graph-conditioned learning across domains** | A single graph-conditioned learning interface applied to per-instance QAOA initialization and to biomedical risk prediction on patient-similarity graphs. | [`src/`](src), [`experiments/`](experiments), [`notebooks/`](notebooks) |

---

## Principal contribution: UQ-QAOA — Uncertainty-Calibrated Trust Regions for Query-Efficient QAOA

> **Manuscript:** *Uncertainty-Calibrated Trust Regions for Query-Efficient QAOA Parameter Search* — [`submission/main.tex`](submission/main.tex).
> **Code artifact (deterministically reproducible):** [`submission/code/`](submission/code).

For low-depth QAOA the dominant practical cost is the number of objective-function *queries* required to identify useful variational parameters. UQ-QAOA repurposes a graph neural network's *predictive covariance* as the metric that defines the local search geometry, rather than as a confidence score.

### Problem setting

For a graph $G=(V,E)$ the MaxCut cost Hamiltonian is

$$
H_C(G)=\sum_{(i,j)\in E}\tfrac{1-Z_iZ_j}{2},
\qquad
\lvert\psi_G(\theta)\rangle=\prod_{\ell=1}^{p}e^{-i\beta_\ell H_M}\,e^{-i\gamma_\ell H_C(G)}\,\lvert+\rangle^{\otimes n},
$$

with mixer $H_M=\sum_i X_i$ and objective $f_G(\theta)=\langle\psi_G(\theta)\lvert H_C(G)\rvert\psi_G(\theta)\rangle$. Cost is measured in objective queries $Q$ (the number of distinct $\theta$ evaluated), and performance is reported as the best-so-far approximation ratio $Q\mapsto\max_{q\le Q} r_G(\theta_q)$, with $r_G=f_G/C_G^\star$.

### Method

A Graph Isomorphism Network (GIN) with spectral positional features predicts a diagonal Gaussian over the QAOA angles,

$$
q_\phi(\theta\mid G)=\mathcal N\!\big(\mu_\phi(G),\,\Sigma_\phi(G)\big),
\qquad
\Sigma_\phi(G)=\mathrm{diag}\big(\sigma_{\min}^2+\mathrm{softplus}(a_{\phi}(G))\big),
$$

trained by Gaussian negative log-likelihood with weight decay. Four information sources—the GIN predictor, a local $k{=}5$ nearest-neighbour prior, a global population prior, and a trotterized quantum-annealing (TQA) schedule—are fused by **precision-weighted (inverse-variance) combination**:

$$
\Sigma_{\mathrm{post}}^{-1}={\Sigma_{\mathrm{GIN}}'}^{-1}+\Sigma_{\mathrm{loc}}^{-1}+\Sigma_{\mathrm{glob}}^{-1}+\Sigma_{\mathrm{TQA}}^{-1},
\qquad
\mu_{\mathrm{post}}=\Sigma_{\mathrm{post}}\!\!\sum_{s}\Sigma_s^{-1}\mu_s .
$$

The posterior covariance induces an anisotropic trust region $\mathcal T_{\phi,\rho}(G)=\{\theta:(\theta-\mu_\phi)^\top(\Sigma_\phi+\lambda I)^{-1}(\theta-\mu_\phi)\le\rho^2\}$ and per-coordinate step sizes $\delta_j=\mathrm{clip}\big(0.5\sqrt{[\Sigma_{\mathrm{post}}]_{jj}},\,0.05,\,0.30\big)$. The search proceeds in three budget-accounted phases: a **TQA safety prefix** (dominance-preserving relative to the strongest physics-informed baseline), deterministic posterior-anchor evaluation, and **sequential greedy coordinate refinement** with trust-region contraction (step halving on stagnation).

### Scope of the guarantees

A best-of-$K$ guarantee makes the geometric intuition precise: when the proposal places mass $\alpha_G(\varepsilon)$ on the $\varepsilon$-optimal subset of the trust region, $K\ge\log(1/\delta)/\alpha_G(\varepsilon)$ samples suffice for $f_G(\widehat\theta_K)\ge f_G^{\mathcal T}-\varepsilon$ with probability $1-\delta$; a finite-shot extension adds a sub-Gaussian union bound, and a conformal construction yields finite-sample coverage under exchangeability. The guarantees are local and conditional by design: they formalize the regime in which a calibrated, compact region saves queries, and they remain distinct from global QAOA optimality.

### Headline result (controlled exact-statevector benchmark, $n{=}14$, $p{=}3$, $Q{=}18$ matched queries)

Mean approximation ratio (higher is better), with 95% bootstrap confidence interval and paired difference relative to TQA, transcribed from [`submission/code/tables/table01_computational_efficiency.csv`](submission/code/tables/table01_computational_efficiency.csv):

| Method | Mean approx. ratio ↑ | 95% bootstrap CI | $\Delta$ vs. TQA |
|---|---|---|---|
| Random | 0.754 | [0.719, 0.788] | −0.100 |
| Heuristic | 0.608 | [0.568, 0.649] | −0.246 |
| $k$-NN | 0.642 | [0.605, 0.676] | −0.211 |
| GNN point | 0.643 | [0.598, 0.689] | −0.211 |
| TQA | 0.853 | [0.829, 0.879] | +0.000 |
| **UQ-QAOA (ours)** | **0.865** | **[0.839, 0.892]** | **+0.012** |

UQ-QAOA exceeds TQA on **7 of 8** held-out instances (paired advantage $+0.012$, 95% bootstrap CI $[+0.003,+0.024]$ over 10,000 resamples) and remains best-or-tied-best at every intermediate query budget. Differential evolution, GP-EI Bayesian optimization, CMA-ES, multi-seed random, and Nelder–Mead each reach at most $0.754$ under $Q{=}18$. The ablation isolates the largest single contribution to the local $k$-NN prior: removing it lowers the ratio to 0.645 (see [`submission/code/tables/table03_ablation.csv`](submission/code/tables/table03_ablation.csv)).

**Higher-powered replication.** On a $6\times$ larger held-out set of **48 instances** (12 per family, same protocol), the advantage holds at **+0.012** with a threefold tighter interval **[+0.008, +0.016]** and a **39/48** win rate — UQ-QAOA 0.862 versus TQA 0.850. This confirmation is generated by [`submission/code/table01_expanded.py`](submission/code/table01_expanded.py) and rendered by [`submission/code/fig05_expanded_benchmark.py`](submission/code/fig05_expanded_benchmark.py) (see [`tables/paired_uq_vs_tqa_expanded.csv`](submission/code/tables/paired_uq_vs_tqa_expanded.csv)).

A C++20 reference backend documents the evaluator contract and records a $4.30\times$ speedup at 8 threads for batched $n{=}14$ evaluation, identifying the mixer kernel as the dominant per-layer cost ($2.60\times$ the cost-phase kernel; see [`tables/bench_thread_scaling.csv`](submission/code/tables/bench_thread_scaling.csv) and [`tables/bench_cpu_kernels.csv`](submission/code/tables/bench_cpu_kernels.csv)). The Python and C++ statevector implementations agree to a maximum absolute error of $5.3\times10^{-13}$ across all tested sizes ([`results/python_cpp_validation.csv`](submission/code/results/python_cpp_validation.csv)).

The reported margin is configuration-dependent rather than a state-of-the-art claim. A controlled reassessment in the manuscript shows that granting the TQA baseline the same greedy coordinate-refinement budget used by UQ-QAOA reverses the ordering on the tested instances; UQ-QAOA is therefore framed as a method study of uncertainty-as-search-geometry.

### Reproduce the principal results end-to-end

```bash
cd submission/code
pip install uq-qaoa                          # installs the uqqaoa-reproduce entry point
# or, from the artifact checkout: pip install .

uqqaoa-reproduce                             # regenerate every figure and table (~2–5 min on the reference platform)
uqqaoa-reproduce --smoke                     # fast end-to-end sanity pass
```

Equivalent direct invocation: `python generate_all.py`. Every output is deterministic given the global seed `260424803`. The mapping from each manuscript figure and table to its generating script is listed in the manuscript's *Reproducibility manifest* and in [`submission/code/README.md`](submission/code/README.md).

---

## Broader work: graph-conditioned learning across quantum optimization and biomedicine

This part of the repository applies the same modeling interface—graph in, calibrated parameterization out—in two domains.

- **Transcriptomic QAOA initialization.** A graph-conditioned GNN predicts depth-2 QAOA angles $(\gamma_1,\gamma_2,\beta_1,\beta_2)$ for MaxCut on prostate transcriptomic co-expression graphs. It reaches a **0.8682** held-out mean approximation ratio against **0.8686** for direct classical search (Nelder–Mead), while reducing median inference from 675.9 ms to **0.256 ms** (≈ 2640× faster) and improving the prior learned baseline (0.8208) by +0.0474 absolute. Source: [`notebooks/qaoa_demo.ipynb`](notebooks/qaoa_demo.ipynb).
- **Cardiotocography (CTG) screening.** A residual clinical GCN on a patient-similarity graph attains **98.8%** accuracy, **0.942** balanced accuracy, and **0.978** ROC AUC on a held-out split ($n=426$, 35 pathologic), improving the simpler graph baseline by +2.1 points accuracy and matching the strongest tabular models on false-positive count. Source: [`notebooks/bio_demo.ipynb`](notebooks/bio_demo.ipynb).

The contribution is bounded by design: it is a unified, transferable graph-learning framework with strong held-out performance, stated independently of universal superiority over every classical baseline.

| Metric | Better | Meaning |
|---|---|---|
| Approximation ratio | higher | fraction of optimal MaxCut value recovered |
| Balanced accuracy | higher | mean recall across classes (informative under class imbalance) |
| ROC AUC | higher | probability a pathologic case is ranked above a normal case |

---

## Interactive notebooks

| Notebook | Role | Contents |
|---|---|---|
| [`notebooks/quantum_ai_bio_combined.ipynb`](notebooks/quantum_ai_bio_combined.ipynb) | Integrated analysis | Shared graph-conditioned formulation spanning both branches |
| [`notebooks/qaoa_demo.ipynb`](notebooks/qaoa_demo.ipynb) | QAOA analysis | Transcriptomic graphs, depth-2 statevector simulation, initializer comparison, ablations |
| [`notebooks/bio_demo.ipynb`](notebooks/bio_demo.ipynb) | Biomedical analysis | Split-first preprocessing, $k$-NN graph construction, graph-versus-tabular evaluation |

Static HTML renders reside in [`website/notebooks_html/`](website/notebooks_html) and are surfaced on the project website.

---

## Repository layout

```text
submission/        UQ-QAOA manuscript (main.tex) and code/ artifact
  code/            reproducible Python package (python/uq_qaoa/), C++20 backend (cpp/),
                   figure/table/experiment scripts, configs, results, tests, reproduce.sh
notebooks/         core analyses and demonstration notebooks (broader work)
experiments/       baseline scripts and extracted evaluations (broader work)
src/               models, simulators, utilities, and serving code (broader work)
data/              source biomedical and transcriptomic inputs
outputs/           processed datasets, tables, and generated results
website/           static-site assets, exported notebook HTML, and paper PDF
index.html         project landing page (GitHub Pages entry point)
```

---

## Reproducibility (notebooks and baselines)

```bash
pip install -r requirements.txt

# Notebooks
jupyter notebook notebooks/quantum_ai_bio_combined.ipynb
jupyter notebook notebooks/qaoa_demo.ipynb
jupyter notebook notebooks/bio_demo.ipynb

# Extracted baselines
python experiments/qaoa/run_qaoa_baselines.py
python experiments/biomedical/run_bio_baselines.py
```

### Local website and prediction demo

The landing page [`index.html`](index.html) includes an optional live QAOA-angle prediction demo backed by a small Flask service.

```bash
# 1) Start the prediction API from the repository root
FLASK_APP=src.server flask run --host=0.0.0.0 --port=5000
# 2) Serve the website
python -m http.server 8000
# 3) Open http://localhost:8000/index.html
```

`website/demo.js` posts to `http://localhost:5000/predict` by default; set `window.API_BASE_URL` before loading it to target another endpoint.

---

## Citation

```bibtex
@misc{huynh2026uqqaoa,
  author       = {Huynh, Molena},
  title        = {Uncertainty-Calibrated Trust Regions for Query-Efficient QAOA Parameter Search},
  year         = {2026},
  howpublished = {Code and manuscript},
  note         = {North Carolina State University; molena.huynh@jmp.com},
  url          = {https://github.com/thmolena/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization}
}
```

## License

Released under the [MIT License](LICENSE).
