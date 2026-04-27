"""Transcriptomic QAOA benchmark construction and noisy evaluation helpers."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import optim

from src.gnn import SimpleGCN


DEFAULT_CACHE_PANEL = Path("data/prostate_top32_variance_panel.csv.gz")
DEFAULT_CACHE_META = Path("data/prostate_top32_variance_panel_meta.json")
LEGACY_CACHE_PANEL = Path("data/prostate_top10_variance_panel.csv.gz")
LEGACY_CACHE_META = Path("data/prostate_top10_variance_panel_meta.json")


@dataclass(frozen=True)
class TranscriptomicBenchmarkConfig:
    top_gene_count: int = 10
    target_edge_count: int = 18
    benchmark_size: int = 6
    benchmark_seed: int = 42
    adaptation_size: int = 24
    adaptation_seed: int = 200
    subsample_size: int = 60
    depth: int = 2
    num_starts: int = 8
    maxiter: int = 320
    seed_offset: int = 1000
    training_seed: int = 7


def headline_transcriptomic_benchmark_config() -> TranscriptomicBenchmarkConfig:
    return TranscriptomicBenchmarkConfig(
        top_gene_count=16,
        target_edge_count=18,
        benchmark_size=6,
        benchmark_seed=42,
        adaptation_size=24,
        adaptation_seed=200,
        subsample_size=60,
        depth=2,
        num_starts=8,
        maxiter=320,
        training_seed=19,
    )


def headline_training_kwargs() -> Dict[str, object]:
    return {
        "hidden_dim": 64,
        "epochs": 800,
        "lr": 2.5e-3,
        "weight_decay": 5e-5,
        "patience": 100,
        "seed": 19,
    }


def classical_search_method_name(depth: int) -> str:
    return f"Classical depth-{int(depth)} search"


def classical_search_angles_method_name(depth: int) -> str:
    return f"Classical depth-{int(depth)} search angles"


def _load_cached_panel(
    panel_path: Path = DEFAULT_CACHE_PANEL,
    meta_path: Path = DEFAULT_CACHE_META,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, object]]:
    candidate_pairs = [(panel_path, meta_path)]
    if (panel_path, meta_path) != (LEGACY_CACHE_PANEL, LEGACY_CACHE_META):
        candidate_pairs.append((LEGACY_CACHE_PANEL, LEGACY_CACHE_META))

    selected_panel_path = None
    selected_meta_path = None
    for current_panel_path, current_meta_path in candidate_pairs:
        if current_panel_path.exists() and current_meta_path.exists():
            selected_panel_path = current_panel_path
            selected_meta_path = current_meta_path
            break

    if selected_panel_path is None or selected_meta_path is None:
        raise FileNotFoundError(
            "Cached transcriptomic panel not found. Expected either data/prostate_top32_variance_panel.csv.gz "
            "and data/prostate_top32_variance_panel_meta.json, or the legacy top-10 cache pair."
        )

    with selected_meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    panel = pd.read_csv(selected_panel_path, compression="gzip")
    sample_ids = panel.pop("__sample_id__").astype(str)
    labels = pd.Series(panel.pop("__target__").astype(str).to_numpy(), index=sample_ids, name=meta.get("label_name", "class"))
    expression_frame = panel.copy()
    expression_frame.index = sample_ids
    gene_table = pd.DataFrame(meta["gene_table"])
    meta["cache_panel_path"] = str(selected_panel_path)
    meta["cache_meta_path"] = str(selected_meta_path)
    return expression_frame, labels, gene_table, meta


def build_cut_diagonal(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Return the cut value for every computational basis state."""
    cut_diagonal = np.zeros(2**n_qubits, dtype=np.float64)
    for state_index in range(2**n_qubits):
        bits = [(state_index >> bit) & 1 for bit in range(n_qubits)]
        cut_diagonal[state_index] = sum(bits[u] != bits[v] for u, v in edges)
    return cut_diagonal


def apply_rx_all(state: np.ndarray, n_qubits: int, beta: float) -> np.ndarray:
    rx = np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=np.complex128,
    )
    psi = state.reshape((2,) * n_qubits)
    for axis in range(n_qubits):
        psi = np.moveaxis(psi, axis, 0)
        psi = np.tensordot(rx, psi, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)


def qaoa_state_fast(cut_diagonal: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    num_states = cut_diagonal.shape[0]
    n_qubits = int(np.log2(num_states))
    state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * cut_diagonal)
        state = apply_rx_all(state, n_qubits, beta)
    return state


def expected_cut_fast(cut_diagonal: np.ndarray, state: np.ndarray) -> float:
    return float(np.dot(cut_diagonal, np.abs(state) ** 2))


def brute_force_maxcut(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    best_cut = -1
    best_mask = 0
    for mask in range(1, 2**n_qubits):
        cut_value = sum(1 for u, v in edges if ((mask >> u) & 1) != ((mask >> v) & 1))
        if cut_value > best_cut:
            best_cut = cut_value
            best_mask = mask
    return best_cut, best_mask


def normalize_angles(raw_angles: Sequence[float], depth: int) -> Tuple[np.ndarray, np.ndarray]:
    raw_angles = np.asarray(raw_angles, dtype=np.float64).reshape(-1)
    if raw_angles.size < 2 * depth:
        raise ValueError(f"Expected at least {2 * depth} raw angles, received {raw_angles.size}.")
    gammas = np.mod(raw_angles[:depth], math.pi)
    betas = np.mod(raw_angles[depth : 2 * depth], math.pi / 2)
    return gammas, betas


def qaoa_value_for_angles(cut_diagonal: np.ndarray, gammas: Sequence[float], betas: Sequence[float]) -> Tuple[float, np.ndarray]:
    state = qaoa_state_fast(cut_diagonal, gammas, betas)
    return expected_cut_fast(cut_diagonal, state), state


def classical_optimize_instance(
    instance: Dict[str, object],
    depth: int,
    num_starts: int = 8,
    maxiter: int = 320,
    seed: int = 0,
) -> Dict[str, object]:
    cut_diagonal = instance["cut_diagonal"]
    rng = np.random.default_rng(seed)
    best = None
    started_at = time.perf_counter()

    def objective(raw_angles: np.ndarray) -> float:
        gammas, betas = normalize_angles(raw_angles, depth)
        value, _ = qaoa_value_for_angles(cut_diagonal, gammas, betas)
        return -value

    for _ in range(num_starts):
        x0 = np.concatenate(
            [
                rng.uniform(0.0, math.pi, size=depth),
                rng.uniform(0.0, math.pi / 2, size=depth),
            ]
        )
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6},
        )
        gammas, betas = normalize_angles(result.x, depth)
        value, state = qaoa_value_for_angles(cut_diagonal, gammas, betas)
        candidate = {
            "gammas": gammas,
            "betas": betas,
            "value": value,
            "state": state,
            "nit": result.nit,
            "nfev": result.nfev,
            "success": bool(result.success),
            "raw_angles": np.concatenate([gammas, betas]),
        }
        if best is None or candidate["value"] > best["value"]:
            best = candidate
    best["runtime_ms"] = 1000.0 * (time.perf_counter() - started_at)
    return best


def select_gene_panel(expression_frame: pd.DataFrame, top_gene_count: int) -> pd.DataFrame:
    gene_table = (
        expression_frame.var(axis=0)
        .sort_values(ascending=False)
        .head(top_gene_count)
        .rename("variance")
        .reset_index()
        .rename(columns={"index": "gene"})
    )
    gene_table["rank"] = np.arange(1, len(gene_table) + 1)
    return gene_table[["rank", "gene", "variance"]]


def build_gene_correlation_graph(
    expression_frame: pd.DataFrame,
    gene_table: pd.DataFrame,
    target_edge_count: int,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    genes = gene_table["gene"].tolist()
    correlation_matrix = expression_frame[genes].corr().abs().fillna(0.0).copy()
    correlation_values = correlation_matrix.to_numpy(copy=True)
    np.fill_diagonal(correlation_values, 0.0)
    correlation_matrix.iloc[:, :] = correlation_values

    complete_graph = nx.Graph()
    for gene_index, gene_name in enumerate(genes):
        complete_graph.add_node(gene_index, gene=gene_name)
    for i, gene_i in enumerate(genes):
        for j in range(i + 1, len(genes)):
            gene_j = genes[j]
            complete_graph.add_edge(i, j, weight=float(correlation_matrix.loc[gene_i, gene_j]))

    spanning_tree = nx.maximum_spanning_tree(complete_graph, weight="weight")
    remaining_edges = sorted(
        (
            (u, v, data["weight"])
            for u, v, data in complete_graph.edges(data=True)
            if not spanning_tree.has_edge(u, v)
        ),
        key=lambda item: item[2],
        reverse=True,
    )

    graph = nx.Graph()
    for node_index, gene_name in enumerate(genes):
        graph.add_node(node_index, gene=gene_name)
    for u, v, data in spanning_tree.edges(data=True):
        graph.add_edge(u, v, weight=data["weight"])
    for u, v, weight in remaining_edges:
        if graph.number_of_edges() >= target_edge_count:
            break
        graph.add_edge(u, v, weight=weight)

    edge_table = pd.DataFrame(
        [
            {
                "gene_u": graph.nodes[u]["gene"],
                "gene_v": graph.nodes[v]["gene"],
                "abs_correlation": data["weight"],
            }
            for u, v, data in graph.edges(data=True)
        ]
    ).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return graph, correlation_matrix, edge_table


def stratified_subsample_indices(labels: pd.Series, sample_size: int, seed: int) -> List[str]:
    label_counts = labels.value_counts().sort_index()
    desired = label_counts / label_counts.sum() * sample_size
    counts = np.floor(desired).astype(int)
    remainder = sample_size - int(counts.sum())
    if remainder > 0:
        fractional = (desired - counts).sort_values(ascending=False)
        for label in fractional.index[:remainder]:
            counts.loc[label] += 1

    rng = np.random.default_rng(seed)
    chosen: List[str] = []
    for label, count in counts.items():
        label_indices = labels[labels == label].index.to_numpy()
        chosen.extend(rng.choice(label_indices, size=int(count), replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def create_graph_instance(
    graph_id: int,
    graph: nx.Graph,
    sample_indices: Sequence[str],
    labels: pd.Series,
    split_name: str,
) -> Dict[str, object]:
    n_qubits = graph.number_of_nodes()
    edges = list(graph.edges())
    best_cut, best_mask = brute_force_maxcut(n_qubits, edges)
    adjacency = nx.to_numpy_array(graph, dtype=np.float64) + np.eye(n_qubits)
    features = adjacency.sum(axis=1, keepdims=True).astype(np.float32)
    return {
        "graph_id": graph_id,
        "split": split_name,
        "graph": graph,
        "n": n_qubits,
        "edges": edges,
        "edge_count": len(edges),
        "density": nx.density(graph),
        "adjacency": adjacency,
        "features": features,
        "cut_diagonal": build_cut_diagonal(n_qubits, edges),
        "best_cut": best_cut,
        "best_mask": best_mask,
        "sample_indices": list(sample_indices),
        "sample_count": len(sample_indices),
        "class_balance": pd.Series(labels.loc[list(sample_indices)]).value_counts().sort_index().to_dict(),
        "gene_labels": [graph.nodes[node]["gene"] for node in graph.nodes()],
    }


def build_graph_split(
    expression_frame: pd.DataFrame,
    labels: pd.Series,
    gene_table: pd.DataFrame,
    target_edge_count: int,
    split_name: str,
    split_size: int,
    subsample_size: int,
    base_seed: int,
) -> List[Dict[str, object]]:
    instances = []
    for offset in range(split_size):
        graph_seed = base_seed + offset
        sample_indices = stratified_subsample_indices(labels, subsample_size, seed=graph_seed)
        subset_expression = expression_frame.loc[sample_indices]
        graph, corr_matrix, edge_table = build_gene_correlation_graph(subset_expression, gene_table, target_edge_count)
        instance = create_graph_instance(graph_seed, graph, sample_indices, labels, split_name)
        instance["correlation_matrix"] = corr_matrix
        instance["edge_table"] = edge_table
        instances.append(instance)
    return instances


def build_transcriptomic_benchmark(
    config: TranscriptomicBenchmarkConfig | None = None,
    panel_path: Path = DEFAULT_CACHE_PANEL,
    meta_path: Path = DEFAULT_CACHE_META,
) -> Dict[str, object]:
    config = config or TranscriptomicBenchmarkConfig()
    expression_frame, labels, gene_table, meta = _load_cached_panel(panel_path, meta_path)
    if len(gene_table) != config.top_gene_count:
        gene_table = select_gene_panel(expression_frame, config.top_gene_count)

    representative_graph, representative_corr, representative_edge_table = build_gene_correlation_graph(
        expression_frame,
        gene_table,
        config.target_edge_count,
    )
    representative = create_graph_instance(
        graph_id=0,
        graph=representative_graph,
        sample_indices=expression_frame.index.tolist(),
        labels=labels,
        split_name="representative",
    )
    representative["correlation_matrix"] = representative_corr
    representative["edge_table"] = representative_edge_table

    adaptation_instances = build_graph_split(
        expression_frame,
        labels,
        gene_table,
        config.target_edge_count,
        "adaptation",
        config.adaptation_size,
        config.subsample_size,
        config.adaptation_seed,
    )
    benchmark_instances = build_graph_split(
        expression_frame,
        labels,
        gene_table,
        config.target_edge_count,
        "benchmark",
        config.benchmark_size,
        config.subsample_size,
        config.benchmark_seed,
    )
    return {
        "config": config,
        "meta": meta,
        "expression_frame": expression_frame,
        "labels": labels,
        "gene_table": gene_table,
        "representative": representative,
        "adaptation_instances": adaptation_instances,
        "benchmark_instances": benchmark_instances,
    }


def attach_classical_targets(instances: Sequence[Dict[str, object]], config: TranscriptomicBenchmarkConfig) -> List[Dict[str, object]]:
    enriched_instances = []
    for index, instance in enumerate(instances):
        reference = classical_optimize_instance(
            instance,
            depth=config.depth,
            num_starts=config.num_starts,
            maxiter=config.maxiter,
            seed=config.seed_offset + int(instance["graph_id"]) + index,
        )
        enriched = dict(instance)
        enriched["classical_reference"] = reference
        enriched["target_angles"] = np.concatenate([reference["gammas"], reference["betas"]]).astype(np.float32)
        enriched_instances.append(enriched)
    return enriched_instances


def train_adapted_qaoa_gnn(
    train_instances: Sequence[Dict[str, object]],
    depth: int,
    hidden_dim: int = 64,
    epochs: int = 500,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    seed: int = 7,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_model = SimpleGCN(in_feats=1, hidden=hidden_dim, out_feats=2, p=depth)
    optimizer = optim.Adam(trained_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(trained_model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        trained_model.train()
        running_loss = 0.0
        for instance in train_instances:
            adjacency_tensor = torch.tensor(instance["adjacency"], dtype=torch.float32)
            feature_tensor = torch.tensor(instance["features"], dtype=torch.float32)
            target_tensor = torch.tensor(instance["target_angles"], dtype=torch.float32)

            prediction = trained_model(feature_tensor, adjacency_tensor).view(-1)
            loss = ((prediction - target_tensor) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        mean_loss = running_loss / max(1, len(train_instances))
        loss_history.append(mean_loss)
        if mean_loss + 1e-8 < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = copy.deepcopy(trained_model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    trained_model.load_state_dict(best_state)
    trained_model.eval()
    return {
        "model": trained_model,
        "history": loss_history,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(loss_history),
    }


def predict_instance_with_gnn(instance: Dict[str, object], model: SimpleGCN, depth: int) -> Dict[str, object]:
    adjacency_tensor = torch.tensor(instance["adjacency"], dtype=torch.float32)
    feature_tensor = torch.tensor(instance["features"], dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(feature_tensor, adjacency_tensor).view(-1).cpu().numpy()
    gammas, betas = normalize_angles(raw_output, depth)
    value, state = qaoa_value_for_angles(instance["cut_diagonal"], gammas, betas)
    return {
        "raw_output": raw_output,
        "gammas": gammas,
        "betas": betas,
        "value": value,
        "state": state,
    }


def predict_instance_with_gnn_overrides(
    instance: Dict[str, object],
    model: SimpleGCN,
    depth: int,
    adjacency_override: np.ndarray | None = None,
    feature_override: np.ndarray | None = None,
) -> Dict[str, object]:
    adjacency = adjacency_override if adjacency_override is not None else instance["adjacency"]
    features = feature_override if feature_override is not None else instance["features"]
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(feature_tensor, adjacency_tensor).view(-1).cpu().numpy()
    gammas, betas = normalize_angles(raw_output, depth)
    value, state = qaoa_value_for_angles(instance["cut_diagonal"], gammas, betas)
    return {
        "raw_output": raw_output,
        "gammas": gammas,
        "betas": betas,
        "value": value,
        "state": state,
    }


def graph_descriptor(instance: Dict[str, object]) -> np.ndarray:
    adjacency = np.asarray(instance["adjacency"], dtype=np.float64)
    n_qubits = adjacency.shape[0]
    weighted_adjacency = adjacency.copy()
    np.fill_diagonal(weighted_adjacency, 0.0)
    edge_weights = weighted_adjacency[np.triu_indices(n_qubits, k=1)]
    nonzero_edge_weights = edge_weights[edge_weights > 0.0]
    if nonzero_edge_weights.size == 0:
        nonzero_edge_weights = np.array([0.0], dtype=np.float64)

    degrees = weighted_adjacency.sum(axis=1)
    features = np.asarray(instance["features"], dtype=np.float64)
    row_means = features.mean(axis=1)
    row_stds = features.std(axis=1)
    return np.array(
        [
            float(instance["n"]),
            float(instance["edge_count"]),
            float(nonzero_edge_weights.mean()),
            float(nonzero_edge_weights.std(ddof=0)),
            float(degrees.mean()),
            float(degrees.std(ddof=0)),
            float(degrees.min()),
            float(degrees.max()),
            float(features.mean()),
            float(features.std(ddof=0)),
            float(row_means.std(ddof=0)),
            float(row_stds.mean()),
        ],
        dtype=np.float64,
    )


def identity_adjacency(instance: Dict[str, object]) -> np.ndarray:
    return np.eye(int(instance["n"]), dtype=np.float32)


def constant_features(instance: Dict[str, object]) -> np.ndarray:
    return np.ones_like(np.asarray(instance["features"], dtype=np.float32), dtype=np.float32)


def build_prior_style_regressor(train_instances: Sequence[Dict[str, object]], seed: int = 19):
    descriptors = np.vstack([graph_descriptor(instance) for instance in train_instances])
    targets = np.vstack([np.asarray(instance["target_angles"], dtype=np.float64) for instance in train_instances])
    regressor = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="tanh",
            solver="lbfgs",
            alpha=1e-3,
            max_iter=4000,
            random_state=seed,
        ),
    )
    regressor.fit(descriptors, targets)
    return regressor


def build_descriptor_knn_regressor(train_instances: Sequence[Dict[str, object]]):
    descriptors = np.vstack([graph_descriptor(instance) for instance in train_instances])
    targets = np.vstack([np.asarray(instance["target_angles"], dtype=np.float64) for instance in train_instances])
    regressor = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=min(3, len(train_instances)), weights="distance"),
    )
    regressor.fit(descriptors, targets)
    return regressor


def evaluate_angle_initializer(
    instance: Dict[str, object],
    method: str,
    raw_angles: np.ndarray,
    proposal_ms: float,
    depth: int,
) -> Dict[str, object]:
    evaluation_started_at = time.perf_counter()
    gammas, betas = normalize_angles(raw_angles, depth)
    value, _ = qaoa_value_for_angles(instance["cut_diagonal"], gammas, betas)
    evaluation_ms = 1000.0 * (time.perf_counter() - evaluation_started_at)
    best_cut = float(instance["best_cut"])
    classical_ratio = float(instance["classical_reference"]["value"] / best_cut)
    ratio = float(value / best_cut)
    return {
        "method": method,
        "graph_id": int(instance["graph_id"]),
        "num_nodes": int(instance["n"]),
        "num_edges": int(instance["edge_count"]),
        "expected_cut": float(value),
        "best_cut": best_cut,
        "approximation_ratio": ratio,
        "classical_ratio": classical_ratio,
        "retention_vs_classical": float(ratio / classical_ratio),
        "proposal_ms": float(proposal_ms),
        "evaluation_ms": float(evaluation_ms),
        "total_ms": float(proposal_ms + evaluation_ms),
    }


def tqa_linear_ramp_raw_angles(depth: int, total_time: float) -> np.ndarray:
    if depth <= 0:
        raise ValueError("QAOA depth must be positive for TQA initialization.")
    fractions = (np.arange(depth, dtype=np.float64) + 0.5) / float(depth)
    dt = float(total_time) / float(depth)
    gammas = fractions * dt
    betas = (1.0 - fractions) * dt
    return np.concatenate([gammas, betas])


def calibrate_tqa_total_time(
    adaptation_instances: Sequence[Dict[str, object]],
    depth: int,
    total_time_grid: Sequence[float] | None = None,
) -> Dict[str, object]:
    if not adaptation_instances:
        raise ValueError("TQA calibration requires at least one adaptation instance.")
    grid = np.asarray(
        list(total_time_grid) if total_time_grid is not None else np.linspace(0.5, 8.0, 31),
        dtype=np.float64,
    )
    if grid.size == 0:
        raise ValueError("TQA calibration grid must contain at least one candidate total time.")

    best_total_time = float(grid[0])
    best_mean_ratio = -float("inf")
    rows: List[Dict[str, float]] = []
    for total_time in grid:
        raw_angles = tqa_linear_ramp_raw_angles(depth, float(total_time))
        ratios = []
        for instance in adaptation_instances:
            gammas, betas = normalize_angles(raw_angles, depth)
            value, _ = qaoa_value_for_angles(instance["cut_diagonal"], gammas, betas)
            ratios.append(float(value / float(instance["best_cut"])))
        mean_ratio = float(np.mean(ratios))
        rows.append(
            {
                "total_time": float(total_time),
                "mean_ratio": mean_ratio,
            }
        )
        if mean_ratio > best_mean_ratio:
            best_mean_ratio = mean_ratio
            best_total_time = float(total_time)

    return {
        "best_total_time": best_total_time,
        "best_mean_ratio": best_mean_ratio,
        "grid_scores": pd.DataFrame(rows),
    }


def budgeted_refine_angles(
    instance: Dict[str, object],
    raw_angles: np.ndarray,
    depth: int,
    maxfev: int,
    maxiter: int | None = None,
    xatol: float = 1e-4,
    fatol: float = 1e-6,
) -> Dict[str, object]:
    if maxfev <= 0:
        raise ValueError("Matched-budget refinement requires a positive maxfev.")

    cut_diagonal = instance["cut_diagonal"]
    x0 = np.asarray(raw_angles, dtype=np.float64).reshape(-1)
    started_at = time.perf_counter()

    def objective(candidate_raw_angles: np.ndarray) -> float:
        gammas, betas = normalize_angles(candidate_raw_angles, depth)
        value, _ = qaoa_value_for_angles(cut_diagonal, gammas, betas)
        return -value

    options = {
        "maxfev": int(maxfev),
        "xatol": float(xatol),
        "fatol": float(fatol),
    }
    if maxiter is not None:
        options["maxiter"] = int(maxiter)

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options=options,
    )
    gammas, betas = normalize_angles(result.x, depth)
    value, _ = qaoa_value_for_angles(cut_diagonal, gammas, betas)
    runtime_ms = 1000.0 * (time.perf_counter() - started_at)
    return {
        "raw_angles": np.concatenate([gammas, betas]),
        "gammas": gammas,
        "betas": betas,
        "value": float(value),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "runtime_ms": float(runtime_ms),
    }


def evaluate_budgeted_initializer(
    instance: Dict[str, object],
    method: str,
    raw_angles: np.ndarray,
    proposal_ms: float,
    depth: int,
    maxfev: int,
    maxiter: int | None = None,
) -> Dict[str, object]:
    initial_gammas, initial_betas = normalize_angles(raw_angles, depth)
    initial_value, _ = qaoa_value_for_angles(instance["cut_diagonal"], initial_gammas, initial_betas)
    refinement = budgeted_refine_angles(
        instance,
        raw_angles,
        depth=depth,
        maxfev=maxfev,
        maxiter=maxiter,
    )
    best_cut = float(instance["best_cut"])
    classical_ratio = float(instance["classical_reference"]["value"] / best_cut)
    refined_ratio = float(refinement["value"] / best_cut)
    initial_ratio = float(initial_value / best_cut)
    return {
        "method": method,
        "budget_evals": int(maxfev),
        "graph_id": int(instance["graph_id"]),
        "num_nodes": int(instance["n"]),
        "num_edges": int(instance["edge_count"]),
        "initial_expected_cut": float(initial_value),
        "expected_cut": float(refinement["value"]),
        "best_cut": best_cut,
        "initial_ratio": initial_ratio,
        "approximation_ratio": refined_ratio,
        "classical_ratio": classical_ratio,
        "retention_vs_classical": float(refined_ratio / classical_ratio),
        "proposal_ms": float(proposal_ms),
        "refinement_ms": float(refinement["runtime_ms"]),
        "total_ms": float(proposal_ms + refinement["runtime_ms"]),
        "nfev": int(refinement["nfev"]),
        "nit": int(refinement["nit"]),
        "success": bool(refinement["success"]),
    }


def summarize_matched_budget_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["used_full_budget"] = annotated["nfev"] >= annotated["budget_evals"]
    summary = (
        annotated.groupby(["budget_evals", "method"], as_index=False)
        .agg(
            num_nodes=("num_nodes", "median"),
            num_edges=("num_edges", "median"),
            mean_initial_ratio=("initial_ratio", "mean"),
            mean_ratio=("approximation_ratio", "mean"),
            std_ratio=("approximation_ratio", "std"),
            mean_retention=("retention_vs_classical", "mean"),
            mean_nfev=("nfev", "mean"),
            budget_hit_rate=("used_full_budget", "mean"),
            median_proposal_ms=("proposal_ms", "median"),
            median_total_ms=("total_ms", "median"),
        )
        .reset_index(drop=True)
    )
    summary["std_ratio"] = summary["std_ratio"].fillna(0.0)
    summary["budget_hit_rate"] = summary["budget_hit_rate"].fillna(0.0)
    method_order = [
        "Heuristic initialization",
        "TQA linear-ramp initialization",
        "GNN-point predictor",
    ]
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    summary = summary.sort_values(["budget_evals", "method"]).reset_index(drop=True)
    return summary


def evaluate_transcriptomic_matched_budget_benchmark(
    benchmark_instances: Sequence[Dict[str, object]],
    model: SimpleGCN,
    adaptation_instances: Sequence[Dict[str, object]],
    depth: int,
    budgets: Sequence[int],
    random_seed: int = 19,
    tqa_total_time_grid: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    del random_seed
    heuristic_raw_angles = np.mean([instance["target_angles"] for instance in adaptation_instances], axis=0)
    tqa_calibration = calibrate_tqa_total_time(
        adaptation_instances,
        depth,
        total_time_grid=tqa_total_time_grid,
    )
    tqa_raw_angles = tqa_linear_ramp_raw_angles(depth, tqa_calibration["best_total_time"])
    rows: List[Dict[str, object]] = []

    normalized_budgets = sorted({int(budget) for budget in budgets if int(budget) > 0})
    if not normalized_budgets:
        raise ValueError("Matched-budget benchmark requires at least one positive evaluation budget.")

    for instance in benchmark_instances:
        gnn_started_at = time.perf_counter()
        gnn_prediction = predict_instance_with_gnn(instance, model, depth)
        gnn_proposal_ms = 1000.0 * (time.perf_counter() - gnn_started_at)

        for budget in normalized_budgets:
            heuristic_started_at = time.perf_counter()
            heuristic_raw = np.asarray(heuristic_raw_angles, dtype=np.float64).copy()
            heuristic_proposal_ms = 1000.0 * (time.perf_counter() - heuristic_started_at)
            rows.append(
                evaluate_budgeted_initializer(
                    instance,
                    "Heuristic initialization",
                    heuristic_raw,
                    heuristic_proposal_ms,
                    depth,
                    maxfev=budget,
                )
            )

            tqa_started_at = time.perf_counter()
            tqa_raw = np.asarray(tqa_raw_angles, dtype=np.float64).copy()
            tqa_proposal_ms = 1000.0 * (time.perf_counter() - tqa_started_at)
            rows.append(
                evaluate_budgeted_initializer(
                    instance,
                    "TQA linear-ramp initialization",
                    tqa_raw,
                    tqa_proposal_ms,
                    depth,
                    maxfev=budget,
                )
            )

            rows.append(
                evaluate_budgeted_initializer(
                    instance,
                    "GNN-point predictor",
                    gnn_prediction["raw_output"],
                    gnn_proposal_ms,
                    depth,
                    maxfev=budget,
                )
            )

    detailed = pd.DataFrame(rows)
    summary = summarize_matched_budget_benchmark(detailed)
    metadata = {
        "tqa_best_total_time": float(tqa_calibration["best_total_time"]),
        "tqa_best_mean_ratio": float(tqa_calibration["best_mean_ratio"]),
        "tqa_grid_scores": tqa_calibration["grid_scores"],
        "budgets": normalized_budgets,
        "note": (
            "This executable code path currently exposes matched-budget comparisons for heuristic, TQA, and "
            "GNN-point initializers only. The manuscript's UQ-QAOA trust-region policy is not implemented in "
            "repository source, so it is intentionally excluded from this runner."
        ),
    }
    return detailed, summary, metadata


def run_transcriptomic_matched_budget_benchmark(
    config: TranscriptomicBenchmarkConfig | None = None,
    training_kwargs: Dict[str, object] | None = None,
    budgets: Sequence[int] = (20, 40, 80),
    tqa_total_time_grid: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    config = config or headline_transcriptomic_benchmark_config()
    bundle = build_transcriptomic_benchmark(config)
    adaptation_instances = attach_classical_targets(bundle["adaptation_instances"], config)
    benchmark_instances = attach_classical_targets(bundle["benchmark_instances"], config)

    fit_kwargs = headline_training_kwargs()
    if training_kwargs:
        fit_kwargs.update(training_kwargs)
    training_result = train_adapted_qaoa_gnn(adaptation_instances, depth=config.depth, **fit_kwargs)

    detailed, summary, matched_budget_meta = evaluate_transcriptomic_matched_budget_benchmark(
        benchmark_instances,
        training_result["model"],
        adaptation_instances,
        config.depth,
        budgets=budgets,
        random_seed=int(fit_kwargs.get("seed", 19)),
        tqa_total_time_grid=tqa_total_time_grid,
    )
    metadata = {
        "config": config,
        "training": {
            "best_loss": training_result["best_loss"],
            "best_epoch": training_result["best_epoch"],
            "epochs_run": training_result["epochs_run"],
        },
        "training_kwargs": fit_kwargs,
        "matched_budget": matched_budget_meta,
    }
    return detailed, summary, metadata


def summarize_initializer_benchmark(frame: pd.DataFrame, depth: int) -> pd.DataFrame:
    classical_method = classical_search_method_name(depth)
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            num_nodes=("num_nodes", "median"),
            num_edges=("num_edges", "median"),
            mean_ratio=("approximation_ratio", "mean"),
            std_ratio=("approximation_ratio", "std"),
            mean_retention=("retention_vs_classical", "mean"),
            median_proposal_ms=("proposal_ms", "median"),
            median_total_ms=("total_ms", "median"),
        )
        .reset_index(drop=True)
    )
    summary["std_ratio"] = summary["std_ratio"].fillna(0.0)
    method_order = [
        classical_method,
        "Random initialization",
        "Heuristic initialization",
        "Descriptor k-NN regressor",
        "Prior-style graph-feature regressor",
        "GNN without graph edges",
        "GNN without node features",
        "Graph-conditioned GNN (ours)",
    ]
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)
    classical_runtime = float(summary.loc[summary["method"] == classical_method, "median_total_ms"].iloc[0])
    full_model_ratio = float(summary.loc[summary["method"] == "Graph-conditioned GNN (ours)", "mean_ratio"].iloc[0])
    summary["speedup_vs_classical"] = classical_runtime / summary["median_total_ms"]
    summary["delta_vs_full"] = summary["mean_ratio"] - full_model_ratio
    return summary


def evaluate_transcriptomic_initializer_benchmark(
    benchmark_instances: Sequence[Dict[str, object]],
    model: SimpleGCN,
    adaptation_instances: Sequence[Dict[str, object]],
    depth: int,
    prior_regressor=None,
    descriptor_knn=None,
    random_seed: int = 19,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classical_method = classical_search_method_name(depth)
    heuristic_raw_angles = np.mean([instance["target_angles"] for instance in adaptation_instances], axis=0)
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, object]] = []

    for instance in benchmark_instances:
        best_cut = float(instance["best_cut"])
        classical_reference = instance["classical_reference"]
        rows.append(
            {
                "method": classical_method,
                "graph_id": int(instance["graph_id"]),
                "num_nodes": int(instance["n"]),
                "num_edges": int(instance["edge_count"]),
                "expected_cut": float(classical_reference["value"]),
                "best_cut": best_cut,
                "approximation_ratio": float(classical_reference["value"] / best_cut),
                "classical_ratio": float(classical_reference["value"] / best_cut),
                "retention_vs_classical": 1.0,
                "proposal_ms": float(classical_reference.get("runtime_ms", np.nan)),
                "evaluation_ms": 0.0,
                "total_ms": float(classical_reference.get("runtime_ms", np.nan)),
            }
        )

        random_started_at = time.perf_counter()
        random_raw_angles = np.concatenate(
            [
                rng.uniform(0.0, math.pi, size=depth),
                rng.uniform(0.0, math.pi / 2, size=depth),
            ]
        )
        random_proposal_ms = 1000.0 * (time.perf_counter() - random_started_at)
        rows.append(
            evaluate_angle_initializer(
                instance,
                "Random initialization",
                random_raw_angles,
                random_proposal_ms,
                depth,
            )
        )

        heuristic_started_at = time.perf_counter()
        heuristic_raw = np.asarray(heuristic_raw_angles, dtype=np.float64).copy()
        heuristic_proposal_ms = 1000.0 * (time.perf_counter() - heuristic_started_at)
        rows.append(
            evaluate_angle_initializer(
                instance,
                "Heuristic initialization",
                heuristic_raw,
                heuristic_proposal_ms,
                depth,
            )
        )

        descriptor = graph_descriptor(instance).reshape(1, -1)
        if descriptor_knn is not None:
            knn_started_at = time.perf_counter()
            knn_raw = descriptor_knn.predict(descriptor).reshape(-1)
            knn_proposal_ms = 1000.0 * (time.perf_counter() - knn_started_at)
            rows.append(
                evaluate_angle_initializer(
                    instance,
                    "Descriptor k-NN regressor",
                    knn_raw,
                    knn_proposal_ms,
                    depth,
                )
            )

        if prior_regressor is not None:
            prior_started_at = time.perf_counter()
            prior_raw = prior_regressor.predict(descriptor).reshape(-1)
            prior_proposal_ms = 1000.0 * (time.perf_counter() - prior_started_at)
            rows.append(
                evaluate_angle_initializer(
                    instance,
                    "Prior-style graph-feature regressor",
                    prior_raw,
                    prior_proposal_ms,
                    depth,
                )
            )

        edge_started_at = time.perf_counter()
        edge_prediction = predict_instance_with_gnn_overrides(
            instance,
            model,
            depth,
            adjacency_override=identity_adjacency(instance),
        )
        edge_proposal_ms = 1000.0 * (time.perf_counter() - edge_started_at)
        rows.append(
            evaluate_angle_initializer(
                instance,
                "GNN without graph edges",
                edge_prediction["raw_output"],
                edge_proposal_ms,
                depth,
            )
        )

        feature_started_at = time.perf_counter()
        feature_prediction = predict_instance_with_gnn_overrides(
            instance,
            model,
            depth,
            feature_override=constant_features(instance),
        )
        feature_proposal_ms = 1000.0 * (time.perf_counter() - feature_started_at)
        rows.append(
            evaluate_angle_initializer(
                instance,
                "GNN without node features",
                feature_prediction["raw_output"],
                feature_proposal_ms,
                depth,
            )
        )

        full_started_at = time.perf_counter()
        full_prediction = predict_instance_with_gnn(instance, model, depth)
        full_proposal_ms = 1000.0 * (time.perf_counter() - full_started_at)
        rows.append(
            evaluate_angle_initializer(
                instance,
                "Graph-conditioned GNN (ours)",
                full_prediction["raw_output"],
                full_proposal_ms,
                depth,
            )
        )

    detailed = pd.DataFrame(rows)
    summary = summarize_initializer_benchmark(detailed, depth)
    return detailed, summary


def run_transcriptomic_headline_benchmark(
    config: TranscriptomicBenchmarkConfig | None = None,
    training_kwargs: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    config = config or headline_transcriptomic_benchmark_config()
    bundle = build_transcriptomic_benchmark(config)
    adaptation_instances = attach_classical_targets(bundle["adaptation_instances"], config)
    benchmark_instances = attach_classical_targets(bundle["benchmark_instances"], config)

    fit_kwargs = headline_training_kwargs()
    if training_kwargs:
        fit_kwargs.update(training_kwargs)
    training_result = train_adapted_qaoa_gnn(adaptation_instances, depth=config.depth, **fit_kwargs)
    prior_regressor = build_prior_style_regressor(adaptation_instances, seed=int(fit_kwargs.get("seed", 19)))
    descriptor_knn = build_descriptor_knn_regressor(adaptation_instances)
    detailed, summary = evaluate_transcriptomic_initializer_benchmark(
        benchmark_instances,
        training_result["model"],
        adaptation_instances,
        config.depth,
        prior_regressor=prior_regressor,
        descriptor_knn=descriptor_knn,
        random_seed=int(fit_kwargs.get("seed", 19)),
    )
    metadata = {
        "config": config,
        "training": {
            "best_loss": training_result["best_loss"],
            "best_epoch": training_result["best_epoch"],
            "epochs_run": training_result["epochs_run"],
        },
        "training_kwargs": fit_kwargs,
    }
    return detailed, summary, metadata


def evaluate_transfer_methods(
    benchmark_instances: Sequence[Dict[str, object]],
    depth: int,
    source_model: SimpleGCN,
    source_heuristic_angles: np.ndarray,
    source_prior_regressor,
    target_model: SimpleGCN | None = None,
) -> pd.DataFrame:
    classical_method = classical_search_method_name(depth)
    rows: List[Dict[str, object]] = []
    for instance in benchmark_instances:
        best_cut = float(instance["best_cut"])
        classical_reference = instance["classical_reference"]
        classical_ratio = float(classical_reference["value"] / best_cut)
        rows.append(
            {
                "method": classical_method,
                "graph_id": int(instance["graph_id"]),
                "num_nodes": int(instance["n"]),
                "num_edges": int(instance["edge_count"]),
                "approximation_ratio": classical_ratio,
                "retention_vs_classical": 1.0,
                "runtime_ms": float(classical_reference.get("runtime_ms", np.nan)),
            }
        )

        source_heuristic_started_at = time.perf_counter()
        source_heuristic = np.asarray(source_heuristic_angles, dtype=np.float64).copy()
        source_heuristic_ms = 1000.0 * (time.perf_counter() - source_heuristic_started_at)
        source_heuristic_row = evaluate_angle_initializer(
            instance,
            "Source-family heuristic",
            source_heuristic,
            source_heuristic_ms,
            depth,
        )
        rows.append(
            {
                "method": source_heuristic_row["method"],
                "graph_id": source_heuristic_row["graph_id"],
                "num_nodes": source_heuristic_row["num_nodes"],
                "num_edges": source_heuristic_row["num_edges"],
                "approximation_ratio": source_heuristic_row["approximation_ratio"],
                "retention_vs_classical": source_heuristic_row["retention_vs_classical"],
                "runtime_ms": source_heuristic_row["total_ms"],
            }
        )

        descriptor = graph_descriptor(instance).reshape(1, -1)
        prior_started_at = time.perf_counter()
        prior_raw = source_prior_regressor.predict(descriptor).reshape(-1)
        prior_ms = 1000.0 * (time.perf_counter() - prior_started_at)
        prior_row = evaluate_angle_initializer(
            instance,
            "Source-family descriptor regressor",
            prior_raw,
            prior_ms,
            depth,
        )
        rows.append(
            {
                "method": prior_row["method"],
                "graph_id": prior_row["graph_id"],
                "num_nodes": prior_row["num_nodes"],
                "num_edges": prior_row["num_edges"],
                "approximation_ratio": prior_row["approximation_ratio"],
                "retention_vs_classical": prior_row["retention_vs_classical"],
                "runtime_ms": prior_row["total_ms"],
            }
        )

        cross_started_at = time.perf_counter()
        cross_prediction = predict_instance_with_gnn(instance, source_model, depth)
        cross_ms = 1000.0 * (time.perf_counter() - cross_started_at)
        cross_row = evaluate_angle_initializer(
            instance,
            "Cross-family GNN",
            cross_prediction["raw_output"],
            cross_ms,
            depth,
        )
        rows.append(
            {
                "method": cross_row["method"],
                "graph_id": cross_row["graph_id"],
                "num_nodes": cross_row["num_nodes"],
                "num_edges": cross_row["num_edges"],
                "approximation_ratio": cross_row["approximation_ratio"],
                "retention_vs_classical": cross_row["retention_vs_classical"],
                "runtime_ms": cross_row["total_ms"],
            }
        )

        if target_model is not None:
            target_started_at = time.perf_counter()
            target_prediction = predict_instance_with_gnn(instance, target_model, depth)
            target_ms = 1000.0 * (time.perf_counter() - target_started_at)
            target_row = evaluate_angle_initializer(
                instance,
                "Target-family GNN (oracle)",
                target_prediction["raw_output"],
                target_ms,
                depth,
            )
            rows.append(
                {
                    "method": target_row["method"],
                    "graph_id": target_row["graph_id"],
                    "num_nodes": target_row["num_nodes"],
                    "num_edges": target_row["num_edges"],
                    "approximation_ratio": target_row["approximation_ratio"],
                    "retention_vs_classical": target_row["retention_vs_classical"],
                    "runtime_ms": target_row["total_ms"],
                }
            )

    return pd.DataFrame(rows)


def summarize_transfer_results(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["source_top_gene_count", "target_top_gene_count", "method"], as_index=False)
        .agg(
            num_nodes=("num_nodes", "median"),
            num_edges=("num_edges", "median"),
            mean_ratio=("approximation_ratio", "mean"),
            std_ratio=("approximation_ratio", "std"),
            retention_vs_classical=("retention_vs_classical", "mean"),
            median_runtime_ms=("runtime_ms", "median"),
        )
        .sort_values(["source_top_gene_count", "target_top_gene_count", "method"])
        .reset_index(drop=True)
    )
    summary["std_ratio"] = summary["std_ratio"].fillna(0.0)
    return summary


def run_cross_family_transfer_experiment(
    source_config: TranscriptomicBenchmarkConfig | None = None,
    target_gene_counts: Sequence[int] = (10, 12, 14, 16),
    training_kwargs: Dict[str, object] | None = None,
    include_target_oracle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    source_config = source_config or headline_transcriptomic_benchmark_config()
    source_bundle = build_transcriptomic_benchmark(source_config)
    source_adaptation = attach_classical_targets(source_bundle["adaptation_instances"], source_config)
    fit_kwargs = headline_training_kwargs()
    if training_kwargs:
        fit_kwargs.update(training_kwargs)
    source_training = train_adapted_qaoa_gnn(source_adaptation, depth=source_config.depth, **fit_kwargs)
    source_prior_regressor = build_prior_style_regressor(source_adaptation, seed=int(fit_kwargs.get("seed", 19)))
    source_heuristic_angles = np.mean([instance["target_angles"] for instance in source_adaptation], axis=0)

    detailed_frames: List[pd.DataFrame] = []
    metadata_targets: List[Dict[str, object]] = []
    for target_gene_count in target_gene_counts:
        target_config = TranscriptomicBenchmarkConfig(
            top_gene_count=int(target_gene_count),
            target_edge_count=target_edge_count_for_gene_count(int(target_gene_count)),
            benchmark_size=source_config.benchmark_size,
            benchmark_seed=source_config.benchmark_seed,
            adaptation_size=source_config.adaptation_size,
            adaptation_seed=source_config.adaptation_seed,
            subsample_size=source_config.subsample_size,
            depth=source_config.depth,
            num_starts=source_config.num_starts,
            maxiter=source_config.maxiter,
            training_seed=source_config.training_seed,
        )
        target_bundle = build_transcriptomic_benchmark(target_config)
        target_benchmark = attach_classical_targets(target_bundle["benchmark_instances"], target_config)
        target_training = None
        if include_target_oracle:
            target_adaptation = attach_classical_targets(target_bundle["adaptation_instances"], target_config)
            target_training = train_adapted_qaoa_gnn(target_adaptation, depth=target_config.depth, **fit_kwargs)
        target_frame = evaluate_transfer_methods(
            target_benchmark,
            target_config.depth,
            source_training["model"],
            source_heuristic_angles,
            source_prior_regressor,
            target_model=None if target_training is None else target_training["model"],
        )
        target_frame.insert(0, "target_top_gene_count", int(target_gene_count))
        target_frame.insert(0, "source_top_gene_count", int(source_config.top_gene_count))
        detailed_frames.append(target_frame)
        metadata_targets.append(
            {
                "target_top_gene_count": int(target_gene_count),
                "target_edge_count": int(target_config.target_edge_count),
                "target_training_best_loss": None if target_training is None else float(target_training["best_loss"]),
                "target_training_best_epoch": None if target_training is None else int(target_training["best_epoch"]),
                "target_training_epochs_run": None if target_training is None else int(target_training["epochs_run"]),
            }
        )

    detailed = pd.concat(detailed_frames, ignore_index=True)
    summary = summarize_transfer_results(detailed)
    metadata = {
        "source_config": source_config,
        "training_kwargs": fit_kwargs,
        "source_training": {
            "best_loss": source_training["best_loss"],
            "best_epoch": source_training["best_epoch"],
            "epochs_run": source_training["epochs_run"],
        },
        "targets": metadata_targets,
    }
    return detailed, summary, metadata


def target_edge_count_for_gene_count(top_gene_count: int, target_density: float = 0.4) -> int:
    if top_gene_count < 2:
        raise ValueError("top_gene_count must be at least 2")
    max_edges = top_gene_count * (top_gene_count - 1) // 2
    suggested_edges = int(round(target_density * max_edges))
    return min(max(top_gene_count - 1, suggested_edges), max_edges)


def evaluate_transcriptomic_benchmark(
    benchmark_instances: Sequence[Dict[str, object]],
    model: SimpleGCN,
    adaptation_instances: Sequence[Dict[str, object]],
    depth: int,
) -> pd.DataFrame:
    classical_method = classical_search_method_name(depth)
    heuristic_angles = np.mean([instance["target_angles"] for instance in adaptation_instances], axis=0)
    rows: List[Dict[str, object]] = []

    for instance in benchmark_instances:
        graph_id = int(instance["graph_id"])
        best_cut = float(instance["best_cut"])

        inference_started_at = time.perf_counter()
        learned = predict_instance_with_gnn(instance, model, depth)
        learned_runtime_ms = 1000.0 * (time.perf_counter() - inference_started_at)

        heuristic_started_at = time.perf_counter()
        heuristic_gammas, heuristic_betas = normalize_angles(heuristic_angles, depth)
        heuristic_value, _ = qaoa_value_for_angles(instance["cut_diagonal"], heuristic_gammas, heuristic_betas)
        heuristic_runtime_ms = 1000.0 * (time.perf_counter() - heuristic_started_at)

        classical_reference = instance["classical_reference"]
        rows.extend(
            [
                {
                    "method": classical_method,
                    "graph_id": graph_id,
                    "num_nodes": int(instance["n"]),
                    "num_edges": int(instance["edge_count"]),
                    "expected_cut": float(classical_reference["value"]),
                    "best_cut": best_cut,
                    "approximation_ratio": float(classical_reference["value"] / best_cut),
                    "runtime_ms": float(classical_reference.get("runtime_ms", np.nan)),
                },
                {
                    "method": "Heuristic mean-angle initializer",
                    "graph_id": graph_id,
                    "num_nodes": int(instance["n"]),
                    "num_edges": int(instance["edge_count"]),
                    "expected_cut": float(heuristic_value),
                    "best_cut": best_cut,
                    "approximation_ratio": float(heuristic_value / best_cut),
                    "runtime_ms": heuristic_runtime_ms,
                },
                {
                    "method": "Graph-conditioned GNN (ours)",
                    "graph_id": graph_id,
                    "num_nodes": int(instance["n"]),
                    "num_edges": int(instance["edge_count"]),
                    "expected_cut": float(learned["value"]),
                    "best_cut": best_cut,
                    "approximation_ratio": float(learned["value"] / best_cut),
                    "runtime_ms": learned_runtime_ms,
                },
            ]
        )

    return pd.DataFrame(rows)


def summarize_transcriptomic_benchmark(frame: pd.DataFrame, depth: int) -> pd.DataFrame:
    classical_method = classical_search_method_name(depth)
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            num_nodes=("num_nodes", "median"),
            num_edges=("num_edges", "median"),
            mean_ratio=("approximation_ratio", "mean"),
            std_ratio=("approximation_ratio", "std"),
            median_runtime_ms=("runtime_ms", "median"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    summary["std_ratio"] = summary["std_ratio"].fillna(0.0)

    classical_mean_ratio = float(
        summary.loc[summary["method"] == classical_method, "mean_ratio"].iloc[0]
    )
    summary["retention_vs_classical"] = summary["mean_ratio"] / classical_mean_ratio
    return summary


def run_transcriptomic_generalization_benchmark(
    config: TranscriptomicBenchmarkConfig | None = None,
    training_kwargs: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    config = config or TranscriptomicBenchmarkConfig()
    bundle = build_transcriptomic_benchmark(config)
    adaptation_instances = attach_classical_targets(bundle["adaptation_instances"], config)
    benchmark_instances = attach_classical_targets(bundle["benchmark_instances"], config)

    fit_kwargs = {
        "depth": config.depth,
        "seed": config.training_seed,
    }
    if training_kwargs:
        fit_kwargs.update(training_kwargs)
    training_result = train_adapted_qaoa_gnn(adaptation_instances, **fit_kwargs)

    benchmark_frame = evaluate_transcriptomic_benchmark(
        benchmark_instances,
        training_result["model"],
        adaptation_instances,
        config.depth,
    )
    summary = summarize_transcriptomic_benchmark(benchmark_frame, config.depth)
    metadata = {
        "config": config,
        "training": {
            "best_loss": training_result["best_loss"],
            "best_epoch": training_result["best_epoch"],
            "epochs_run": training_result["epochs_run"],
        },
    }
    return benchmark_frame, summary, metadata


def rx_unitary(beta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=np.complex128,
    )


def apply_single_qubit_unitary_density(
    density: np.ndarray,
    unitary: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    operator = np.array([[1.0 + 0.0j]])
    for index in range(n_qubits):
        operator = np.kron(operator, unitary if index == qubit else identity)
    return operator @ density @ operator.conj().T


def apply_local_depolarizing(density: np.ndarray, error_rate: float, n_qubits: int) -> np.ndarray:
    if error_rate <= 0.0:
        return density
    paulis = [
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    mixed = density
    for qubit in range(n_qubits):
        updated = (1.0 - error_rate) * mixed
        for pauli in paulis:
            updated += (error_rate / 3.0) * apply_single_qubit_unitary_density(mixed, pauli, qubit, n_qubits)
        mixed = updated
    return mixed


def qaoa_density_with_local_depolarizing(
    cut_diagonal: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    error_rate: float,
) -> np.ndarray:
    state = np.ones(cut_diagonal.shape[0], dtype=np.complex128) / np.sqrt(cut_diagonal.shape[0])
    density = np.outer(state, state.conj())
    n_qubits = int(np.log2(cut_diagonal.shape[0]))
    phase = np.eye(cut_diagonal.shape[0], dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        phase = np.diag(np.exp(-1j * gamma * cut_diagonal))
        density = phase @ density @ phase.conj().T
        density = apply_local_depolarizing(density, error_rate, n_qubits)
        for qubit in range(n_qubits):
            density = apply_single_qubit_unitary_density(density, rx_unitary(beta), qubit, n_qubits)
        density = apply_local_depolarizing(density, error_rate, n_qubits)
    return density


def noisy_expected_cut(cut_diagonal: np.ndarray, density: np.ndarray) -> float:
    diagonal_probabilities = np.real(np.diag(density))
    return float(np.dot(cut_diagonal, diagonal_probabilities))


def evaluate_method_under_noise(
    benchmark_instances: Sequence[Dict[str, object]],
    method_name: str,
    angle_lookup: Dict[int, np.ndarray],
    error_rates: Iterable[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for error_rate in error_rates:
        ratios = []
        for instance in benchmark_instances:
            raw_angles = angle_lookup[int(instance["graph_id"])]
            gammas, betas = normalize_angles(raw_angles, raw_angles.size // 2)
            density = qaoa_density_with_local_depolarizing(instance["cut_diagonal"], gammas, betas, error_rate)
            value = noisy_expected_cut(instance["cut_diagonal"], density)
            ratio = value / float(instance["best_cut"])
            ratios.append(ratio)
            rows.append(
                {
                    "method": method_name,
                    "graph_id": int(instance["graph_id"]),
                    "noise_rate": error_rate,
                    "noisy_value": value,
                    "noisy_ratio": ratio,
                }
            )
        rows.append(
            {
                "method": method_name,
                "graph_id": "mean",
                "noise_rate": error_rate,
                "noisy_value": float(np.mean(ratios)),
                "noisy_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios, ddof=0)),
            }
        )
    return rows


def run_transcriptomic_noise_experiment(
    config: TranscriptomicBenchmarkConfig | None = None,
    noise_rates: Sequence[float] = (0.0, 0.01, 0.02, 0.05),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    config = config or TranscriptomicBenchmarkConfig()
    bundle = build_transcriptomic_benchmark(config)
    adaptation_instances = attach_classical_targets(bundle["adaptation_instances"], config)
    benchmark_instances = attach_classical_targets(bundle["benchmark_instances"], config)
    training_result = train_adapted_qaoa_gnn(adaptation_instances, depth=config.depth, seed=config.training_seed)
    model = training_result["model"]

    learned_lookup: Dict[int, np.ndarray] = {}
    classical_lookup: Dict[int, np.ndarray] = {}
    heuristic_angles = np.mean([instance["target_angles"] for instance in adaptation_instances], axis=0)
    heuristic_lookup: Dict[int, np.ndarray] = {}
    for instance in benchmark_instances:
        graph_id = int(instance["graph_id"])
        prediction = predict_instance_with_gnn(instance, model, config.depth)
        learned_lookup[graph_id] = np.concatenate([prediction["gammas"], prediction["betas"]])
        classical_lookup[graph_id] = instance["target_angles"]
        heuristic_lookup[graph_id] = heuristic_angles

    rows: List[Dict[str, object]] = []
    rows.extend(evaluate_method_under_noise(benchmark_instances, "Graph-conditioned GNN (ours)", learned_lookup, noise_rates))
    rows.extend(
        evaluate_method_under_noise(
            benchmark_instances,
            classical_search_angles_method_name(config.depth),
            classical_lookup,
            noise_rates,
        )
    )
    rows.extend(evaluate_method_under_noise(benchmark_instances, "Heuristic mean-angle initializer", heuristic_lookup, noise_rates))

    frame = pd.DataFrame(rows)
    summary = frame[frame["graph_id"] == "mean"].copy()
    summary = summary[["method", "noise_rate", "noisy_ratio", "std_ratio"]].rename(columns={"noisy_ratio": "mean_ratio"})
    summary = summary.sort_values(["noise_rate", "method"]).reset_index(drop=True)
    metadata = {
        "config": config,
        "training": {
            "best_loss": training_result["best_loss"],
            "best_epoch": training_result["best_epoch"],
            "epochs_run": training_result["epochs_run"],
        },
        "noise_rates": list(noise_rates),
    }
    return summary, metadata
