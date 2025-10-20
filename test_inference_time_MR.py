#!/usr/bin/env python3
"""
test_inference_time_MR.py
────────────────────────────────────────────────────────────────────────────
Benchmark `ModelAPI.predict_action` latency while *synthetically* scaling the
episodic memory bank from 0 × to 4 × its real size.

Key idea
~~~~~~~~
For every bucket we **clone its (angle, outs) pairs three extra times**
and nudge each clone’s angle by a tiny ε so the list remains sorted and
`bisect_left()` still works:

    ε = 1e-4  (wraps around at 1.0)

That gives a realistic traversal cost without altering any categorical keys.
"""

from __future__ import annotations
import pathlib, time, math, random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt           # harmless when PLOT=False

# ──────────────────────────────────────────────────────────────────────────
# 0) Config – edit if you like
# ──────────────────────────────────────────────────────────────────────────
CSV_PATH      = pathlib.Path("RL_test_jun_23.csv")
MODEL_PATH    = pathlib.Path("multihead_classifier_with_meta.pth")
MEM_PATH      = pathlib.Path("memory_bank_dynamic.pkl")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZES = [0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0]       # × original memory
N_SAMPLES = 1_000
SEED       = 42
PLOT       = False                                    # True → save PNG

# ──────────────────────────────────────────────────────────────────────────
# 1) Import inference API
# ──────────────────────────────────────────────────────────────────────────
from model_api import ModelAPI                        # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 2) Helpers
# ──────────────────────────────────────────────────────────────────────────
def count_episodes(mem: Dict[Any, Any]) -> int:
    """Total (angle, outs) pairs in *mem*."""
    return sum(len(b["angles"]) for b in mem.values())


def enlarge_memory(original: Dict[Any, Any],
                   factor: int = 4,
                   eps: float = 1e-4) -> Dict[Any, Any]:
    """
    Return a **new** memory dict whose episode count is `factor` × original.

    Each bucket’s lists are extended by making `factor-1` clones of every
    element, shifting angle by ±k·eps (wrap-around at 1.0) to keep strict
    order and unique bisect positions.
    """
    if factor <= 1:
        # deep-copy to stay immutable
        return {k: {"angles": list(v["angles"]), "outs": list(v["outs"])}
                for k, v in original.items()}

    new_mem: Dict[Any, Any] = {}
    for key, bucket in original.items():
        A, O = bucket["angles"], bucket["outs"]
        new_A: List[float] = []
        new_O: List[Tuple[int, ...]] = []

        for ang, out in zip(A, O):
            # keep original
            new_A.append(ang)
            new_O.append(out)
            # add (factor-1) shifted copies
            for j in range(1, factor):
                shifted = (ang + j * eps) % 1.0
                new_A.append(shifted)
                new_O.append(out)

        # re-sort (angle list may no longer be monotone after wrap)
        order = sorted(range(len(new_A)), key=new_A.__getitem__)
        new_mem[key] = {
            "angles": [new_A[i] for i in order],
            "outs":   [new_O[i] for i in order],
        }
    return new_mem


def prepare_inputs(csv_path: pathlib.Path,
                   feature_order: List[str],
                   labels: Dict[str, np.ndarray],
                   n_rows: int,
                   seed: int) -> np.ndarray:
    """Load CSV and snap every prev_* value to its discrete label (exact dtype)."""
    df = pd.read_csv(csv_path)

    # derive movement_label_norm if absent
    if "movement_label_norm" not in df.columns:
        if "movement_label" not in df.columns:
            raise RuntimeError("CSV needs movement_label_norm or movement_label.")
        df["movement_label_norm"] = df["movement_label"] / 6.0

    # snap prev_* columns
    for col in feature_order[2:]:
        next_col = col.replace("prev_", "next_")
        disc = labels[next_col]
        vals = df[col].astype(disc.dtype).values
        idx = np.abs(vals[:, None] - disc[None, :]).argmin(axis=1)
        df[col] = disc[idx]

    # subset rows deterministically
    X = df[feature_order].values
    if n_rows < len(X):
        rs = np.random.RandomState(seed)
        X = X[rs.choice(len(X), n_rows, replace=False)]
    return X


def benchmark(api: ModelAPI, X: np.ndarray, warm: int = 10) -> float:
    """Return mean latency per sample in milliseconds."""
    for i in range(min(warm, len(X))):
        api.predict_action(X[i])
    if api.device.type == "cuda":
        torch.cuda.synchronize(api.device)
    t0 = time.perf_counter()
    for x in X:
        api.predict_action(x)
    if api.device.type == "cuda":
        torch.cuda.synchronize(api.device)
    t1 = time.perf_counter()
    return (t1 - t0) / len(X) * 1e3


# ──────────────────────────────────────────────────────────────────────────
# 3) Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # load base model + memory
    api_base = ModelAPI.from_files(
        str(MODEL_PATH), str(MEM_PATH), tolerance=0.05, device=DEVICE
    )
    base_n = count_episodes(api_base.memory)
    print(f"Original memory: {base_n:,} episodes")

    # build 4× memory once (deep-copy cost only paid up front)
    mem_4x = enlarge_memory(api_base.memory, factor=4)
    four_n = count_episodes(mem_4x)
    assert four_n == base_n * 4
    print(f"Synthetic 4× memory: {four_n:,} episodes")

    # inputs
    X_bench = prepare_inputs(
        CSV_PATH, api_base.FEATURE_ORDER, api_base.labels,
        N_SAMPLES, SEED
    )
    print(f"Benchmarking on {len(X_bench):,} inputs")

    # sweep
    results: List[Tuple[float, int, float]] = []         # (multiplier, episodes, ms)
    for mult in SIZES:
        if mult == 0.0:
            mem = {}
            eps_cnt = 0
        elif mult <= 1.0:
            # down-sample from *original* memory
            keep = int(round(mult * base_n))
            if keep == 0: keep = 1
            rng = random.Random(SEED)
            from itertools import islice
            # quick uniform sample of indices
            flat = [(k, a, o) for k, b in api_base.memory.items()
                               for a, o in zip(b["angles"], b["outs"])]
            chosen = rng.sample(flat, keep)
            mem = {}
            for k, a, o in chosen:
                b = mem.setdefault(k, {"angles": [], "outs": []})
                b["angles"].append(a); b["outs"].append(o)
            for b in mem.values():                        # re-sort
                order = sorted(range(len(b["angles"])), key=b["angles"].__getitem__)
                b["angles"] = [b["angles"][i] for i in order]
                b["outs"]   = [b["outs"][i]   for i in order]
            eps_cnt = keep
        else:
            # >1 : take subset of 4× memory
            keep = int(round(mult * base_n))
            rng = random.Random(SEED)
            flat = [(k, a, o) for k, b in mem_4x.items()
                               for a, o in zip(b["angles"], b["outs"])]
            chosen = rng.sample(flat, keep)
            mem = {}
            for k, a, o in chosen:
                b = mem.setdefault(k, {"angles": [], "outs": []})
                b["angles"].append(a); b["outs"].append(o)
            for b in mem.values():
                order = sorted(range(len(b["angles"])), key=b["angles"].__getitem__)
                b["angles"] = [b["angles"][i] for i in order]
                b["outs"]   = [b["outs"][i]   for i in order]
            eps_cnt = keep

        api = ModelAPI(
            model           = api_base.model,
            head_cols       = api_base.head_cols,
            discrete_labels = api_base.labels,
            memory_bank     = mem,
            tolerance       = api_base.tol,
            device          = api_base.device,
        )
        ms = benchmark(api, X_bench)
        results.append((mult, eps_cnt, ms))
        print(f"  ×{mult:<4} → {eps_cnt:>8,} eps  :  {ms:6.3f} ms")

    # summary
    print("\n=== Latency vs synthetic memory size ===")
    print(f"{'mult':>4} | {'episodes':>10} | {'mean-ms':>8}")
    print("-" * 29)
    for m, e, t in results:
        print(f"{m:>4.1f} | {e:10,} | {t:8.3f}")

    # optional plot
    if PLOT:
        xs = [e for _, e, _ in results]
        ys = [t for _, _, t in results]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Episodes in memory bank")
        plt.ylabel("Mean latency per call (ms)")
        plt.title("Latency vs synthetic memory size")
        plt.grid(True)
        plt.tight_layout()
        out = pathlib.Path("latency_vs_memory.png")
        plt.savefig(out, dpi=160)
        print(f"\nSaved plot → {out.resolve()}")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
