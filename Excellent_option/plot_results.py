#!/usr/bin/env python
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("Excellent_option", "results")


def load_pytorch(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pairs_from_rows(rows: List[Dict]) -> List[Tuple[int, int]]:
    out = []
    for r in rows:
        key = (int(r["in_features"]), int(r["out_features"]))
        if key not in out:
            out.append(key)
    return out


def _extract_metric(rows: List[Dict], mode: str, metric: str, keys: List[Tuple[int, int]]) -> List[float]:
    by_key = {(int(r["in_features"]), int(r["out_features"])): r for r in rows if r["mode"] == mode}
    return [float(by_key[k][metric]) for k in keys]


def plot_pytorch(p_data: Dict) -> None:
    base = p_data["baseline_tf32_off"]
    tc = p_data["tensorcore_tf32_on"]
    keys = [(r["in_features"], r["out_features"]) for r in base]
    labels = [f"{k}x{n}" for (k, n) in keys]
    x = list(range(len(keys)))

    base_ms = [r["avg_ms"] for r in base]
    tc_ms = [r["avg_ms"] for r in tc]
    base_tflops = [r["tflops"] for r in base]
    tc_tflops = [r["tflops"] for r in tc]
    speedup = [b / t for b, t in zip(base_ms, tc_ms)]

    plt.figure(figsize=(8, 5))
    plt.plot(x, base_ms, marker="o", label="TF32 off (FP32 baseline)")
    plt.plot(x, tc_ms, marker="o", label="TF32 on (Tensor Core path)")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Latency per forward (ms)")
    plt.xlabel("FC size (in_features x out_features)")
    plt.title("PyTorch FC Latency Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pytorch_latency.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, base_tflops, marker="o", label="TF32 off (FP32 baseline)")
    plt.plot(x, tc_tflops, marker="o", label="TF32 on (Tensor Core path)")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Throughput (TFLOP/s)")
    plt.xlabel("FC size (in_features x out_features)")
    plt.title("PyTorch FC Throughput Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pytorch_throughput.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, speedup, marker="o", color="purple")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Speedup (x)")
    plt.xlabel("FC size (in_features x out_features)")
    plt.title("PyTorch TF32-on Speedup vs TF32-off")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pytorch_speedup.png"), dpi=160)
    plt.close()


def plot_cublas(rows: List[Dict]) -> None:
    keys = _pairs_from_rows(rows)
    labels = [f"{k}x{n}" for (k, n) in keys]
    x = list(range(len(keys)))

    base_ms = _extract_metric(rows, "sgemm", "avg_ms", keys)
    tc_ms = _extract_metric(rows, "tf32_tensor_op", "avg_ms", keys)
    base_tflops = _extract_metric(rows, "sgemm", "tflops", keys)
    tc_tflops = _extract_metric(rows, "tf32_tensor_op", "tflops", keys)
    speedup = [b / t for b, t in zip(base_ms, tc_ms)]

    plt.figure(figsize=(8, 5))
    plt.plot(x, base_ms, marker="o", label="cublasSgemm FP32")
    plt.plot(x, tc_ms, marker="o", label="cublasGemmEx TF32 tensor ops")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Latency per forward (ms)")
    plt.xlabel("FC size (in_features x out_features)")
    plt.title("cuBLAS FC Latency Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cublas_latency.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, base_tflops, marker="o", label="cublasSgemm FP32")
    plt.plot(x, tc_tflops, marker="o", label="cublasGemmEx TF32 tensor ops")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Throughput (TFLOP/s)")
    plt.xlabel("FC size (in_features x out_features)")
    plt.title("cuBLAS FC Throughput Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cublas_throughput.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, speedup, marker="o", color="darkgreen")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Speedup (x)")
    plt.xlabel("FC size (in_features x out_features)")
    plt.title("cuBLAS TF32 TensorOp Speedup vs SGEMM")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cublas_speedup.png"), dpi=160)
    plt.close()


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pytorch_json = os.path.join(RESULTS_DIR, "pytorch_gemm_results.json")
    if os.path.isfile(pytorch_json):
        plot_pytorch(load_pytorch(pytorch_json))
        print(f"Generated PyTorch plots from {pytorch_json}")
    else:
        print(f"Skipping PyTorch plots, missing {pytorch_json}")

    cublas_csv = os.path.join(RESULTS_DIR, "cublas_gemm_results.csv")
    if os.path.isfile(cublas_csv):
        plot_cublas(load_csv_rows(cublas_csv))
        print(f"Generated cuBLAS plots from {cublas_csv}")
    else:
        print(f"Skipping cuBLAS plots, missing {cublas_csv}")

    print(f"Done. Plots are in {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
