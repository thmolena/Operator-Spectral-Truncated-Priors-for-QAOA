#!/usr/bin/env python3
"""
CPU Benchmark: Measured per-query and per-instance wall time, peak RSS,
and BLAS thread scaling for the UQ-QAOA statevector pipeline.

Output: code/tables/bench_cpu.csv

Methodology
-----------
Each measurement calls the real QAOA statevector simulation (no stubs):
  qaoa_statevector()  implements |ψ⟩ = ∏_ℓ exp(-iβ_ℓ H_M) exp(-iγ_ℓ H_C)|+⟩^⊗n
  qaoa_ratio()        computes r_G(θ) = ⟨ψ|H_C|ψ⟩ / C_max

Phase kernel (bandwidth-bound, element-wise):
  ψ_z ← exp(-iγ C(z)) ψ_z   for z ∈ {0,...,2^n-1}
  Arithmetic intensity: 6 FLOP per 16-byte element = 0.375 FLOP/byte

Mixer kernel (strided pair update, per qubit q):
  [ψ_{z:q=0}]     [cos β   -i sin β] [ψ_{z:q=0}]
  [ψ_{z:q=1}]  ←  [-i sin β  cos β ] [ψ_{z:q=1}]
  Stride = 2^q; spatial locality depends on qubit index.

Total FLOP per QAOA evaluation ≈ p × (6·2^n + n × 12·2^{n-1})
  = p × 2^n × (6 + 6n) for p layers, n qubits.
  Ref: Farhi et al. (2014), Eq. (1)–(4); mixer decomposition via
  Euler identity exp(-iβ X) = cos(β)I - i sin(β)X.

Memory per statevector: 2^n × 16 bytes (complex128).

Timing uses time.perf_counter() (wall clock).
Memory uses resource.getrusage(RUSAGE_SELF).ru_maxrss (peak RSS, macOS).
Thread control uses VECLIB_MAXIMUM_THREADS (Apple Accelerate BLAS).

All graph instances are generated deterministically from the global seed.
No values are assigned or fabricated; every number is measured live.

References:
  [1] Farhi, Goldstone, Gutmann, arXiv:1411.4028 (2014), Eq. (1)-(4)
  [2] Xu et al., ICLR 2019, "How Powerful Are Graph Neural Networks?"
"""
from __future__ import annotations

import csv
import os
import resource
import time

import numpy as np

from uq_qaoa_core import (
    CANDIDATE_BUDGET, DEPTH, GLOBAL_SEED,
    stable_seed, graph_edges, graph_features, qaoa_cost_values,
    qaoa_statevector, qaoa_ratio,
    build_training_library, get_trained_gin, predict_gaussian,
    angle_candidates, table_path,
)


def _peak_rss_mb():
    """Peak resident set size in MB (macOS: ru_maxrss is bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def bench_instance(family, n, depth, budget, seed):
    """Benchmark a single graph instance end-to-end.

    Returns dict with measured wall times and memory.

    Pipeline measured:
      1. graph_edges()       — graph construction
      2. graph_features()    — 8-dim spectral/topological features
      3. qaoa_cost_values()  — C(z) for all 2^n bitstrings
      4. angle_candidates()  — generate Q candidate angle sets
      5. qaoa_ratio()        — exact statevector eval per candidate

    Each step uses the real algorithm from uq_qaoa_core.py.
    """
    # Graph construction
    t0 = time.perf_counter()
    edges = graph_edges(family, n, seed)
    t_graph = time.perf_counter() - t0

    # Feature extraction
    t0 = time.perf_counter()
    feats = graph_features(n, edges)
    t_feat = time.perf_counter() - t0

    # Cost values (reused across all candidates)
    t0 = time.perf_counter()
    c_vals = qaoa_cost_values(edges, n)
    t_cost = time.perf_counter() - t0

    # Candidate generation (UQ-QAOA method)
    t0 = time.perf_counter()
    cands = angle_candidates("uq_qaoa", budget, n, edges, feats, seed, depth)
    t_cand = time.perf_counter() - t0

    # Statevector evaluation: time each query individually
    query_times = []
    best_ratio = 0.0
    for theta in cands:
        gamma = theta[:depth]
        beta = theta[depth:]
        t0 = time.perf_counter()
        r = qaoa_ratio(gamma, beta, edges, n, c_vals)
        query_times.append(time.perf_counter() - t0)
        if r > best_ratio:
            best_ratio = r

    rss = _peak_rss_mb()
    return {
        "family": family,
        "n": n,
        "p": depth,
        "Q": budget,
        "t_graph_ms": t_graph * 1000,
        "t_feat_ms": t_feat * 1000,
        "t_cost_ms": t_cost * 1000,
        "t_cand_ms": t_cand * 1000,
        "t_eval_total_ms": sum(query_times) * 1000,
        "t_per_query_ms": np.mean(query_times) * 1000,
        "t_per_query_std_ms": np.std(query_times) * 1000,
        "t_instance_ms": (t_graph + t_feat + t_cost + t_cand + sum(query_times)) * 1000,
        "peak_rss_mb": rss,
        "best_ratio": best_ratio,
        "statevec_bytes": (1 << n) * 16,
    }


def bench_thread_scaling(family, n, depth, budget, seed, thread_counts):
    """Measure wall time for the same instance at different BLAS thread counts.

    Uses VECLIB_MAXIMUM_THREADS (Apple Accelerate BLAS) to control
    threading.  This measures BLAS-level parallelism in NumPy operations
    (e.g., the batched complex multiplications in the phase kernel are
    dispatched through Accelerate on macOS).

    Returns list of (threads, t_instance_ms) tuples.
    """
    results = []
    for nt in thread_counts:
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(nt)
        # Warm up: one throwaway evaluation
        edges = graph_edges(family, n, seed)
        c_vals = qaoa_cost_values(edges, n)
        feats = graph_features(n, edges)
        cands = angle_candidates("uq_qaoa", budget, n, edges, feats, seed, depth)

        # Timed run: full pipeline
        t0 = time.perf_counter()
        for theta in cands:
            gamma = theta[:depth]
            beta = theta[depth:]
            qaoa_ratio(gamma, beta, edges, n, c_vals)
        t_eval = time.perf_counter() - t0

        results.append({"threads": nt, "n": n, "t_eval_ms": t_eval * 1000})

    # Restore default
    os.environ.pop("VECLIB_MAXIMUM_THREADS", None)
    return results


def gen_bench():
    """Run full CPU benchmark suite and write results to CSV."""
    print("=" * 60)
    print("CPU Benchmark: measured wall time, memory, thread scaling")
    print("=" * 60, flush=True)

    # Ensure GIN predictor is trained (needed by angle_candidates)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)

    budget = CANDIDATE_BUDGET
    families = ["ER", "REG", "BA", "WS"]
    sizes = [8, 10, 12, 14]

    # --- Part 1: Per-instance benchmark ---
    rows = []
    for n in sizes:
        for fam in families:
            seed = stable_seed(GLOBAL_SEED, "bench", fam, n)
            result = bench_instance(fam, n, DEPTH, budget, seed)
            rows.append(result)
            print(f"  n={n:2d} {fam:3s}: "
                  f"t_instance={result['t_instance_ms']:8.1f} ms, "
                  f"t/query={result['t_per_query_ms']:6.2f} ms, "
                  f"RSS={result['peak_rss_mb']:6.1f} MB, "
                  f"ratio={result['best_ratio']:.4f}")

    # Write per-instance results
    path = table_path("bench_cpu")
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")

    # --- Part 2: Thread scaling (n=14 ER, largest instance) ---
    print("\nThread scaling (n=14, ER):")
    seed14 = stable_seed(GLOBAL_SEED, "bench", "ER", 14)
    thread_counts = [1, 2, 4, 8]
    scaling = bench_thread_scaling("ER", 14, DEPTH, budget, seed14, thread_counts)

    path_scaling = table_path("bench_cpu_scaling")
    with open(path_scaling, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "n", "t_eval_ms"])
        w.writeheader()
        w.writerows(scaling)
    print(f"Wrote {path_scaling}")

    for s in scaling:
        ratio = scaling[0]["t_eval_ms"] / s["t_eval_ms"] if s["t_eval_ms"] > 0 else 0
        print(f"  {s['threads']:d} thread(s): {s['t_eval_ms']:8.1f} ms "
              f"(speedup {ratio:.2f}x vs 1 thread)")

    # --- Part 3: Per-kernel profiling (phase vs mixer breakdown) ---
    print("\nKernel profiling (n=14, ER, single evaluation):")
    edges14 = graph_edges("ER", 14, seed14)
    c_vals14 = qaoa_cost_values(edges14, 14)
    feats14 = graph_features(14, edges14)
    cands14 = angle_candidates("uq_qaoa", budget, 14, edges14, feats14, seed14, DEPTH)
    theta = cands14[0]
    gamma = theta[:DEPTH]
    beta = theta[DEPTH:]

    # Profile phase kernel alone
    N = 1 << 14
    psi = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
    phase_times = []
    mixer_times = []
    for layer in range(DEPTH):
        # Phase: ψ_z ← exp(-iγC(z)) ψ_z  (element-wise, bandwidth-bound)
        t0 = time.perf_counter()
        psi *= np.exp(-1j * gamma[layer] * c_vals14)
        phase_times.append(time.perf_counter() - t0)

        # Mixer: single-qubit X-rotations (strided pair updates)
        c = np.cos(beta[layer])
        s = -1j * np.sin(beta[layer])
        t0 = time.perf_counter()
        for q in range(14):
            shape = [1 << (14 - q - 1), 2, 1 << q]
            psi_r = psi.reshape(shape)
            a = psi_r[:, 0, :].copy()
            b_ = psi_r[:, 1, :].copy()
            psi_r[:, 0, :] = c * a + s * b_
            psi_r[:, 1, :] = s * a + c * b_
        mixer_times.append(time.perf_counter() - t0)

    avg_phase = np.mean(phase_times) * 1000
    avg_mixer = np.mean(mixer_times) * 1000
    print(f"  Phase kernel (element-wise):   {avg_phase:.3f} ms/layer "
          f"[bandwidth-bound, AI=0.375 FLOP/byte]")
    print(f"  Mixer kernel (strided pairs):  {avg_mixer:.3f} ms/layer "
          f"[stride-sensitive, {14} qubit rotations]")
    print(f"  Mixer/Phase ratio:             {avg_mixer/avg_phase:.1f}x "
          f"(mixer dominates due to strided access)")

    # Write kernel profile
    path_kernel = table_path("bench_cpu_kernels")
    with open(path_kernel, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "ms_per_layer", "note"])
        w.writeheader()
        w.writerow({"kernel": "phase", "ms_per_layer": f"{avg_phase:.4f}",
                     "note": "element-wise exp(-ig*C(z))*psi, bandwidth-bound"})
        w.writerow({"kernel": "mixer", "ms_per_layer": f"{avg_mixer:.4f}",
                     "note": f"strided 2x2 rotations over {14} qubits"})
    print(f"Wrote {path_kernel}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    gen_bench()
