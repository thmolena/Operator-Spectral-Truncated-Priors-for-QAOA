#!/usr/bin/env python3
"""Quick diagnostic: print per-method, per-family performance numbers."""
import numpy as np
import uq_qaoa_core as uc

instances = []
for n in [8, 10, 12, 14]:
    instances.extend(uc.make_instances(uc.FAMILIES, n, 2))

results = uc.eval_all_methods(instances, depth=uc.DEPTH, budget=18)

print("Method                    Mean    Std     Min     Max")
print("-" * 60)
for m, label in zip(uc.METHODS, uc.METHOD_LABELS):
    ratios = [r[0] for r in results[m]]
    arr = np.array(ratios)
    print(f"{label:25s} {arr.mean():.4f}  {arr.std():.4f}  {arr.min():.4f}  {arr.max():.4f}")

print()
print("=== Per-family breakdown (Heuristic vs UQ-QAOA) ===")
for fam in uc.FAMILIES:
    h_vals, u_vals = [], []
    for r_idx, (f, inst, e, feats, n_graph) in enumerate(instances):
        if f == fam:
            h_vals.append(results["heuristic"][r_idx][0])
            u_vals.append(results["uq_qaoa"][r_idx][0])
    print(f"  {fam}: Heuristic={np.mean(h_vals):.4f}, UQ-QAOA={np.mean(u_vals):.4f}, "
          f"diff={np.mean(u_vals)-np.mean(h_vals):.4f}")

print()
print("=== Per-size breakdown (all methods) ===")
sizes = [8, 10, 12, 14]
for n in sizes:
    idxs = [i for i, (f, inst, e, feats, ng) in enumerate(instances) if ng == n]
    print(f"  n={n}:")
    for m, label in zip(uc.METHODS, uc.METHOD_LABELS):
        vals = [results[m][i][0] for i in idxs]
        print(f"    {label:25s} {np.mean(vals):.4f}")
