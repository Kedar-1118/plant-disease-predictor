"""
training/benchmark.py
----------------------
Performance benchmark for the PlantGuard AI inference pipeline.

Measures:
  - Model cold-start load time
  - Per-image inference latency (p50, p95, p99)
  - Two-stage pipeline end-to-end latency

HOW TO RUN:
    python training/benchmark.py --image path/to/leaf.jpg
    python training/benchmark.py --image path/to/leaf.jpg --iterations 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_model_loading():
    """Measure cold-start model loading time."""
    import prediction.predict as pred

    # Reset cached models
    pred._crop_model = None
    pred._disease_models = {}
    pred._crop_classes = None
    pred._disease_classes = {}

    start = time.perf_counter()
    pred._load_crop_model()
    crop_load_time = time.perf_counter() - start

    print(f"  Crop model load time:    {crop_load_time*1000:.0f} ms")

    # Load each disease model
    for crop in pred.get_supported_crops():
        start = time.perf_counter()
        pred._load_disease_model(crop)
        load_time = time.perf_counter() - start
        print(f"  {crop} disease model load: {load_time*1000:.0f} ms")


def benchmark_inference(image_path: str, iterations: int = 50):
    """
    Measure inference latency for the full two-stage pipeline.

    Args:
        image_path:  Path to a test image.
        iterations:  Number of inference runs.
    """
    from prediction.predict import full_pipeline

    # Warm-up run (ensures models are cached)
    print("\n  Warming up (first inference)...")
    warmup_start = time.perf_counter()
    full_pipeline(image_path)
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warm-up (cold start): {warmup_time*1000:.0f} ms")

    # Benchmark runs
    print(f"\n  Running {iterations} iterations...")
    latencies = []

    for i in range(iterations):
        start = time.perf_counter()
        full_pipeline(image_path)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{iterations}] Current: {elapsed*1000:.0f} ms")

    latencies = np.array(latencies)

    print("\n" + "─" * 50)
    print("  BENCHMARK RESULTS")
    print("─" * 50)
    print(f"  Iterations: {iterations}")
    print(f"  Mean:       {np.mean(latencies):.0f} ms")
    print(f"  Std:        {np.std(latencies):.0f} ms")
    print(f"  Min:        {np.min(latencies):.0f} ms")
    print(f"  Max:        {np.max(latencies):.0f} ms")
    print(f"  p50:        {np.percentile(latencies, 50):.0f} ms")
    print(f"  p95:        {np.percentile(latencies, 95):.0f} ms")
    print(f"  p99:        {np.percentile(latencies, 99):.0f} ms")
    print(f"  Throughput: {1000 / np.mean(latencies):.1f} images/sec")
    print("─" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PlantGuard AI inference performance"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to a test leaf image"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--skip-loading",
        action="store_true",
        help="Skip model loading benchmark",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("  PlantGuard AI — Performance Benchmark")
    print("=" * 60)

    if not args.skip_loading:
        print("\n[1/2] Model Loading Benchmark")
        benchmark_model_loading()

    print("\n[2/2] Inference Latency Benchmark")
    benchmark_inference(args.image, args.iterations)

    print("\n[DONE] Benchmark complete!")


if __name__ == "__main__":
    main()
