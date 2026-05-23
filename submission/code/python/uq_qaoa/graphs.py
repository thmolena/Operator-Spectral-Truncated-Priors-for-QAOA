from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import networkx as nx
import numpy as np

SUPPORTED_FAMILIES = {"er", "random_regular", "watts_strogatz", "barabasi_albert", "cycle"}


@dataclass(frozen=True)
class GraphInstance:
    n: int
    edges: tuple[tuple[int, int], ...]
    family: str
    seed: int
    graph_id: str

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def _canonical_edges(g: nx.Graph) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in g.edges() if u != v))


def generate_graph(family: str, n: int, seed: int) -> GraphInstance:
    family = family.lower()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported graph family {family}")
    if n <= 1:
        raise ValueError("n must exceed 1")
    if family == "cycle":
        g = nx.cycle_graph(n)
    elif family == "er":
        g = nx.gnp_random_graph(n, 0.5, seed=seed)
    elif family == "random_regular":
        degree = min(3, n - 1)
        if (n * degree) % 2:
            degree = max(2, degree - 1)
        g = nx.random_regular_graph(degree, n, seed=seed)
    elif family == "watts_strogatz":
        k = min(4, n - 1)
        if k % 2:
            k -= 1
        g = nx.watts_strogatz_graph(n, max(2, k), 0.3, seed=seed)
    else:
        g = nx.barabasi_albert_graph(n, min(2, n - 1), seed=seed)
    edges = _canonical_edges(g)
    return GraphInstance(n=n, edges=edges, family=family, seed=int(seed), graph_id=f"{family}_n{n}_s{seed}")


def save_edge_list(graph: GraphInstance, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"n": graph.n, "family": graph.family, "seed": graph.seed, "edges": graph.edges}, indent=2), encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["u", "v"])
        w.writerows(graph.edges)


def load_edge_list(path: str | Path, family: str = "loaded", seed: int = 0) -> GraphInstance:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        edges = tuple((int(u), int(v)) for u, v in data["edges"])
        return GraphInstance(int(data["n"]), edges, data.get("family", family), int(data.get("seed", seed)), path.stem)
    edges = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            edges.append((int(row["u"]), int(row["v"])))
    n = max(max(e) for e in edges) + 1 if edges else 0
    return GraphInstance(n, tuple(edges), family, seed, path.stem)


def graph_features(graph: GraphInstance) -> np.ndarray:
    g = nx.Graph()
    g.add_nodes_from(range(graph.n))
    g.add_edges_from(graph.edges)
    degrees = np.array([d for _, d in g.degree()], dtype=float)
    density = nx.density(g)
    clustering = nx.average_clustering(g) if graph.n > 2 else 0.0
    lap = nx.laplacian_matrix(g).astype(float).toarray()
    evals = np.linalg.eigvalsh(lap) if graph.n > 0 else np.zeros(1)
    lambda2 = float(evals[1]) if evals.size > 1 else 0.0
    lambdan = float(evals[-1]) if evals.size else 0.0
    return np.array([density, degrees.mean(), degrees.std(), clustering, lambda2, lambdan, graph.n, len(graph.edges)], dtype=float)


def split_seeds(seeds: list[int]) -> dict[str, list[int]]:
    # Deterministic non-overlapping split by position to prevent leakage.
    return {"train": seeds[0::3], "validation": seeds[1::3], "test": seeds[2::3]}
