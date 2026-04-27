# Graph Neural Networks for the Quantum Approximate Optimization Algorithm (QAOA)

### GNN-Based Parameter Prediction, Learned Warm-Start Initialization, and Convergence Analysis — with Applications to Network-Based Biomedical Systems

[![Project Website](https://img.shields.io/badge/Project_Website-Open-0f766e?style=for-the-badge)](https://thmolena.github.io/Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/)
[![Research Paper PDF](https://img.shields.io/badge/Paper-PDF-1d4ed8?style=for-the-badge)](research_paper/main.pdf)

This repository develops **graph neural network (GNN) methods for graph-conditioned QAOA parameter prediction and graph-based biomedical inference**. The central technical idea is that both settings can be cast as learning on structured graphs: transcriptomic co-expression graphs for depth-2 QAOA MaxCut initialization, and cardiotocography patient-similarity graphs for node-level pathologic-risk prediction.

The main result is a unified graph-conditioned learning framework with strong held-out performance in both domains. On transcriptomic MaxCut graphs, the QAOA model reaches near-search approximation quality with a single 0.256 ms forward pass. On cardiotocography, the residual clinical GCN substantially improves simpler graph baselines while remaining competitive with the strongest tabular models. The repository therefore contributes both a practical QAOA initializer and a transferable graph-learning formulation spanning quantum optimization and biomedical decision support.

The practical bottleneck in QAOA is classical parameter search. This repository studies whether graph-conditioned learning can replace repeated per-instance optimization while preserving held-out quality, and whether the same formulation remains useful in a biomedical graph setting. The project is organized around three concrete artifacts:

- transcriptomic co-expression graphs used to predict depth-2 QAOA parameters for maximum cut (MaxCut)
- cardiotocography similarity graphs used to predict node-level pathologic-risk scores
- an integrated notebook and manuscript that place both tasks in the same graph-to-parameterization formulation

## Main Contributions and Comparative Results

This repository makes three claims on fixed held-out splits.

1. A graph-conditioned GNN predicts depth-2 QAOA parameters for transcriptomic MaxCut graphs with near-search quality at orders-of-magnitude lower cost.
2. A residual clinical GCN substantially improves over simpler graph baselines for CTG screening and remains competitive with the strongest tabular models.
3. The same graph-conditioned learning framework transfers across quantum optimization and biomedical risk prediction.

### Metric Conventions

| Metric | Better direction | Interpretation |
|---|---|---|
| Approximation ratio | Higher | Fraction of the optimal MaxCut value recovered; in the delta column, negative means the competing method underperforms this work. |
| Accuracy | Higher | Fraction of held-out exams classified correctly. |
| Balanced accuracy | Higher | Mean recall across classes; essential here because only 35 of 426 test cases are pathologic. |
| ROC AUC | Higher | Probability a pathologic case is ranked above a normal case. |
| TP / 35 | Higher | Pathologic cases detected on the held-out set. |
| FP | Lower | Normal cases incorrectly flagged as pathologic. |

### QAOA Branch — Held-Out Approximation Ratio

Same 6-graph transcriptomic evaluation set throughout.

| Method | Mean approx. ratio | Delta vs. this work | Interpretation |
|---|---|---|---|
| Zero angles | 0.7224 | −0.1458 | No optimization; trivial lower bound. |
| Prior-style learned baseline | 0.8208 | −0.0474 | Learned warm start without graph conditioning. |
| Direct classical search (Nelder-Mead) | 0.8686 | +0.0004 | Near-identical quality, but 675.9 ms per instance. |
| Random search (best of 256 evaluations) | 0.8954 | +0.0272 | Higher quality only after 256 circuit evaluations. |
| Goemans-Williamson SDP | 0.8780 | +0.0098 | Classical polynomial-time reference with 0.878 guarantee. |
| **★ This work: graph-conditioned GNN** | **0.8682** | — | **Single forward pass; 0.256 ms inference.** |

The central QAOA result is the tradeoff, not the raw table entry: this model improves the prior learned baseline by +0.0474 absolute (+5.77% relative), retains 99.95% of direct-search quality, and reduces median inference time from 675.9 ms to 0.256 ms, a 2640x speedup.

### CTG Biomedical Branch — Held-Out Metrics

Same split throughout; test set size $n = 426$.

| Method | Accuracy | Balanced acc. | ROC AUC | TP / 35 | FP |
|---|---|---|---|---|---|
| Logistic Regression | 94.1% | 0.916 | 0.984 | 31 | 21 |
| Random Forest | 96.9% | 0.905 | 0.994 | 29 | 7 |
| MLP | 98.4% | 0.926 | 0.971 | 30 | 2 |
| LightGBM | 98.6% | 0.927 | 0.993 | 30 | 1 |
| XGBoost | 98.8% | 0.955 | 0.992 | 32 | 2 |
| Calibrated LightGBM | **99.1%** | **0.956** | 0.991 | 32 | 1 |
| AdaptiveBioGCN | 96.7% ± 0.97% | — | — | — | — |
| **★ ResidualClinicalGCN** | **98.8%** | **0.942** | **0.978** | **31** | **1** |

The biomedical contribution is not a claim that graph learning surpasses the best tabular model on raw accuracy. The contribution is that the graph model improves the simpler graph baseline by +2.1 percentage points in accuracy and +0.057 in balanced accuracy, matches XGBoost on accuracy, matches the best false-positive count, and adds patient-similarity structure that tabular models do not represent.

### What Is New Here

1. **Graph conditioning materially improves learned QAOA initialization.** The gain over the prior learned baseline is large, while the gap to full classical search is negligible.
2. **The same modeling interface works across both branches.** In one case the output is a QAOA angle vector; in the other it is a node-level clinical risk score.
3. **The CTG graph model adds structural evidence, not just a score.** Predictions are made on a patient-similarity graph rather than from isolated tabular rows.
4. **The claims are bounded.** This repository argues for a unified, transferable graph-learning framework with strong empirical performance, not for universal superiority over every classical baseline.

---

## Visual Overview

<table>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_benchmark_overview.png" alt="Held-out QAOA benchmark overview" />
      <p><strong>Held-out QAOA quality.</strong> Graph-conditioned GNN: <strong>0.8682</strong>. Direct classical search: <strong>0.8686</strong>.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/qaoa_demo_landscape_geometry.png" alt="QAOA landscape geometry analysis" />
      <p><strong>QAOA landscape geometry.</strong> The visible high-value basin is concentrated, which helps explain why learned warm starts reduce search burden on held-out graphs.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="website/notebooks_html/figures/bio_demo_heldout_evaluation.png" alt="Held-out CTG evaluation" />
      <p><strong>Held-out CTG operating point.</strong> ResidualClinicalGCN reaches <strong>98.8%</strong> accuracy and <strong>0.942</strong> balanced accuracy.</p>
    </td>
    <td width="50%">
      <img src="website/notebooks_html/figures/combined_transcriptomic_benchmark.png" alt="Integrated benchmark figure" />
      <p><strong>Shared formulation.</strong> Both branches are cast in one graph-to-parameterization-to-objective pipeline.</p>
    </td>
  </tr>
</table>

## System View

| Branch | Input graph | Learned parameterization | Downstream objective | Main evidence |
| --- | --- | --- | --- | --- |
| Quantum Approximate Optimization Algorithm (QAOA) | prostate transcriptomic co-expression graph | depth-2 angle vector $(\gamma_1, \gamma_2, \beta_1, \beta_2)$ | expected maximum cut (MaxCut) value | held-out approximation ratio, ablations, runtime |
| Biomedical | cardiotocography (CTG) patient-similarity graph | node-level pathologic-risk scores | thresholded screening behavior | accuracy, balanced accuracy, calibration, robustness |
| Integrated | shared graph-conditioned interface | task-specific decision variables | branch-specific downstream evaluation | comparative framing across both domains |

## Analysis Notebooks

| Notebook | Role | Contents |
| --- | --- | --- |
| [notebooks/quantum_ai_bio_combined.ipynb](notebooks/quantum_ai_bio_combined.ipynb) | Integrated analysis | Shared graph-conditioned formulation spanning both branches |
| [notebooks/qaoa_demo.ipynb](notebooks/qaoa_demo.ipynb) | QAOA analysis | Transcriptomic graphs, depth-2 statevector simulation, initializer comparison, and ablations |
| [notebooks/bio_demo.ipynb](notebooks/bio_demo.ipynb) | Biomedical analysis | Split-first preprocessing, k-NN graph construction, and graph-versus-tabular evaluation |

## Representative Results

This section summarizes the takeaways already established above rather than repeating the full comparisons.

| Branch | Key result | Why it matters |
| --- | --- | --- |
| QAOA | Graph-conditioned GNN reaches 0.8682 held-out mean approximation ratio versus 0.8686 for direct classical search | Near-search quality with a single learned forward pass |
| QAOA | Median inference time falls from 675.9 ms to 0.256 ms | 2640x lower per-instance cost |
| QAOA | Prior learned baseline reaches 0.8208 | Graph conditioning adds a clear quality gain over earlier learned initialization |
| CTG | ResidualClinicalGCN reaches 98.8% accuracy, 0.942 balanced accuracy, and 0.978 ROC AUC | Strong graph-model operating point on the held-out clinical split |
| CTG | Calibrated LightGBM reaches 99.06% accuracy and 0.956 balanced accuracy | Best tabular reference remains slightly stronger on raw discrimination metrics |
| Integrated | One graph-conditioned framework supports both QAOA angle prediction and clinical risk scoring | Establishes the repository's main methodological contribution |

## Artifacts

| Artifact | Path | Purpose |
| --- | --- | --- |
| Website entry point | [index.html](index.html) | Project landing page |
| Paper PDF | [research_paper/main.pdf](research_paper/main.pdf) | Manuscript version of the project |
| Notebook HTML exports | [website/notebooks_html](website/notebooks_html) | Static rendered analyses |
| QAOA baselines | [experiments/qaoa/run_qaoa_baselines.py](experiments/qaoa/run_qaoa_baselines.py) | Reproduce QAOA comparison runs |
| Biomedical baselines | [experiments/biomedical/run_bio_baselines.py](experiments/biomedical/run_bio_baselines.py) | Reproduce CTG comparison runs |

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks:

```bash
jupyter notebook notebooks/quantum_ai_bio_combined.ipynb
jupyter notebook notebooks/qaoa_demo.ipynb
jupyter notebook notebooks/bio_demo.ipynb
```

Run the extracted baseline scripts:

```bash
python experiments/qaoa/run_qaoa_baselines.py
python experiments/biomedical/run_bio_baselines.py
```

## Repository Layout

```text
notebooks/      core analyses and demonstration notebooks
experiments/    baseline scripts and extracted evaluations
src/            models, simulators, utilities, and serving code
data/           source biomedical and transcriptomic inputs
outputs/        processed datasets, tables, and generated artifacts
research_paper/ manuscript draft
website/        static site and exported notebook HTML
```

## License

This project is released under the terms of the LICENSE file in this repository.