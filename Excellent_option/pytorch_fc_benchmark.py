#!/usr/bin/env python
import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def parse_sizes(raw: str) -> List[Tuple[int, int]]:
    pairs = []
    for item in raw.split(","):
        left, right = item.strip().split("x")
        pairs.append((int(left), int(right)))
    return pairs


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")
    return torch.device("cuda")


def time_forward(x: torch.Tensor, model: nn.Module, iters: int, warmup: int) -> float:
    device = x.device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms: List[float] = []
        for _ in range(iters):
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
    return sum(times_ms) / len(times_ms)


def run_mode(
    allow_tf32: bool,
    sizes: List[Tuple[int, int]],
    batch_size: int,
    device: torch.device,
    iters: int,
    warmup: int,
) -> List[Dict]:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    mode_rows: List[Dict] = []
    for in_features, out_features in sizes:
        x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
        model = FCModel(in_features, out_features).to(device)
        avg_ms = time_forward(x, model, iters=iters, warmup=warmup)
        flops = 2.0 * batch_size * in_features * out_features
        tflops = flops / (avg_ms * 1e-3) / 1e12
        mode_rows.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "batch_size": batch_size,
                "avg_ms": avg_ms,
                "tflops": tflops,
            }
        )
    return mode_rows


def write_summary_csv(path: str, baseline: List[Dict], tensorcore: List[Dict]) -> None:
    headers = [
        "framework",
        "mode",
        "batch_size",
        "in_features",
        "out_features",
        "avg_ms",
        "tflops",
        "speedup_vs_tf32_off",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        baseline_map = {(r["in_features"], r["out_features"]): r for r in baseline}

        for row in baseline:
            writer.writerow(
                {
                    "framework": "pytorch",
                    "mode": "tf32_off",
                    "batch_size": row["batch_size"],
                    "in_features": row["in_features"],
                    "out_features": row["out_features"],
                    "avg_ms": f'{row["avg_ms"]:.6f}',
                    "tflops": f'{row["tflops"]:.6f}',
                    "speedup_vs_tf32_off": "1.000000",
                }
            )

        for row in tensorcore:
            key = (row["in_features"], row["out_features"])
            base_ms = baseline_map[key]["avg_ms"]
            speedup = base_ms / row["avg_ms"] if row["avg_ms"] > 0.0 else 0.0
            writer.writerow(
                {
                    "framework": "pytorch",
                    "mode": "tf32_on",
                    "batch_size": row["batch_size"],
                    "in_features": row["in_features"],
                    "out_features": row["out_features"],
                    "avg_ms": f'{row["avg_ms"]:.6f}',
                    "tflops": f'{row["tflops"]:.6f}',
                    "speedup_vs_tf32_off": f"{speedup:.6f}",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch FC TF32 benchmark")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--sizes",
        type=str,
        default="1024x1024,2048x2048,4096x4096,8192x8192",
        help='Comma separated list like "1024x1024,2048x2048"',
    )
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    sizes = parse_sizes(args.sizes)
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}, warmup: {args.warmup}, iters: {args.iters}")
    print(f"Sizes: {sizes}")

    print("Running Mode 1: FP32 matmul with TF32 disabled...")
    baseline = run_mode(
        False, sizes, args.batch_size, device=device, iters=args.iters, warmup=args.warmup
    )
    print("Running Mode 2: FP32 matmul with TF32 enabled...")
    tensorcore = run_mode(
        True, sizes, args.batch_size, device=device, iters=args.iters, warmup=args.warmup
    )

    out = {
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "sizes": sizes,
        "baseline_tf32_off": baseline,
        "tensorcore_tf32_on": tensorcore,
    }
    results_dir = os.path.join("Excellent_option", "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "pytorch_gemm_results.json")
    csv_path = os.path.join(results_dir, "pytorch_gemm_results.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_summary_csv(csv_path, baseline, tensorcore)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

