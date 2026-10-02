"""CLI: PCA image compression with custom Power and Jacobi eigen-solvers.

Examples
--------
    python main.py --selftest
    python main.py --image images/sample.png --size 64 --k 5 10 20 40 --method both
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write PNG files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import eigen  # noqa: E402
import pca  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent  # constant path, never modified

# A "solution" is a dict with keys:
#   "values"  : eigenvalues, descending           (1-D array)
#   "vectors" : matching eigenvectors as columns  (2-D array)
#   "count"   : total power iterations OR Jacobi sweeps (int)
#   "seconds" : wall-clock solver time            (float)
Solution = dict[str, Any]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", default=str(BASE_DIR / "images" / "sample.png"),
                        help="path to an image (converted to grayscale)")
    parser.add_argument("--size", type=int, default=64,
                        help="resize to size x size pixels (default 64)")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 40],
                        help="numbers of principal components to keep")
    parser.add_argument("--method", choices=["power", "jacobi", "both"], default="both")
    parser.add_argument("--selftest", action="store_true",
                        help="check both solvers on a 3x3 matrix and exit")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Loading and solving
# --------------------------------------------------------------------------
def load_image(path: str, size: int) -> np.ndarray:
    """Load an image as a size x size float matrix of gray levels in [0, 255]."""
    image = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(image, dtype=float)


def clean_k_values(ks: list[int], size: int) -> list[int]:
    """Sort, de-duplicate, and drop k values that are < 1 or > size (with a notice)."""
    valid = sorted({k for k in ks if 1 <= k <= size})
    dropped = sorted(set(ks) - set(valid))
    if dropped:
        print(f"Note: ignoring k = {dropped} (must be between 1 and {size}).")
    if not valid:
        raise SystemExit("error: no valid k values left")
    return valid


def solve(C: np.ndarray, method: str, k_max: int) -> Solution:
    """Run one eigen-solver on C and time it.

    Power method computes only the top k_max pairs (deflation);
    Jacobi always computes all of them.
    """
    start = time.perf_counter()
    if method == "power":
        values, vectors, iterations = eigen.power_method_with_deflation(C, k_max)
        count = int(iterations.sum())
    else:
        values, vectors, count = eigen.jacobi_eigen(C)
    seconds = time.perf_counter() - start
    return {"values": values, "vectors": vectors, "count": count, "seconds": seconds}


# --------------------------------------------------------------------------
# Validation against numpy (np.linalg.eigh is used ONLY in this function)
# --------------------------------------------------------------------------
def validate_against_eigh(C: np.ndarray, solution: Solution, m: int) -> tuple[float, float, float]:
    """Compare the top m computed eigenpairs with numpy's reference.

    eigenvalue error  = max_i |lambda_i - lambda_i_ref|
    alignment_i       = |v_i . u_i|   (1 means same direction; the absolute
                        value ignores the arbitrary sign of eigenvectors)
    eigenvector error = max_i (1 - alignment_i)

    Returns (max eigenvalue error, eigenvector error, minimum alignment).
    """
    ref_values, ref_vectors = np.linalg.eigh(C)  # ascending order
    ref_values, ref_vectors = ref_values[::-1], ref_vectors[:, ::-1]

    value_error = np.max(np.abs(solution["values"][:m] - ref_values[:m]))
    alignment = np.abs(np.sum(solution["vectors"][:, :m] * ref_vectors[:, :m], axis=0))
    alignment = np.minimum(alignment, 1.0)  # rounding can give 1.0000000000000002
    return float(value_error), float(np.max(1.0 - alignment)), float(np.min(alignment))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def build_comparison_rows(C: np.ndarray, solutions: dict[str, Solution], m: int) -> list[dict[str, Any]]:
    """One row per method with runtime, iterations/sweeps and errors."""
    rows = []
    for method, sol in solutions.items():
        value_err, vector_err, min_align = validate_against_eigh(C, sol, m)
        rows.append({
            "method": method,
            "runtime_s": sol["seconds"],
            "count": sol["count"],
            "count_unit": "iterations" if method == "power" else "sweeps",
            "max_eigenvalue_error": value_err,
            "eigenvector_error": vector_err,
            "min_alignment": min_align,
        })
    return rows


def print_comparison(rows: list[dict[str, Any]], m: int) -> None:
    print(f"\nSolver comparison (top {m} eigenpairs vs np.linalg.eigh)")
    print(f"{'method':<8}{'runtime [s]':>13}{'count':>9} {'unit':<11}"
          f"{'max |dLambda|':>15}{'max 1-|v.u|':>13}{'min |v.u|':>11}")
    for r in rows:
        print(f"{r['method']:<8}{r['runtime_s']:>13.4f}{r['count']:>9d} {r['count_unit']:<11}"
              f"{r['max_eigenvalue_error']:>15.3e}{r['eigenvector_error']:>13.3e}"
              f"{r['min_alignment']:>11.6f}")


def save_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Compression experiment
# --------------------------------------------------------------------------
def evaluate_k(X: np.ndarray, X_centered: np.ndarray, mean: np.ndarray,
               solution: Solution, total_variance: float, k: int) -> dict[str, Any]:
    """Compress with k components, reconstruct, and measure quality."""
    V = solution["vectors"]
    Z = pca.compress(X_centered, V, k)
    X_hat = pca.reconstruct(Z, V, mean, k)
    return {
        "k": k,
        "image": X_hat,
        "mse": pca.mse(X, X_hat),
        "psnr": pca.psnr(X, X_hat),
        "ratio": pca.compression_ratio(X.shape[0], X.shape[1], k),
        "variance": pca.variance_retained(solution["values"], k, total_variance),
    }


def print_metrics(method: str, results: list[dict[str, Any]]) -> None:
    print(f"\nMetrics for method = {method}")
    print(f"{'k':>4}{'MSE':>12}{'PSNR [dB]':>12}{'comp. ratio':>13}{'variance kept':>15}")
    for r in results:
        print(f"{r['k']:>4d}{r['mse']:>12.2f}{r['psnr']:>12.2f}"
              f"{r['ratio']:>13.2f}{r['variance']:>15.4%}")


# --------------------------------------------------------------------------
# Plots (each saves one PNG into outdir)
# --------------------------------------------------------------------------
def plot_reconstructions(X: np.ndarray, results: dict[str, list[dict[str, Any]]], outdir: Path) -> None:
    """Plot 1: original next to the reconstruction for every k; one row per method."""
    n_cols = 1 + len(next(iter(results.values())))
    fig, axes = plt.subplots(len(results), n_cols, figsize=(2.6 * n_cols, 2.8 * len(results)),
                             squeeze=False)
    for row, (method, res) in enumerate(results.items()):
        panels = [("original", X)] + [(f"k = {r['k']}", r["image"]) for r in res]
        for col, (title, img) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(method, fontsize=11)
    fig.suptitle("Original vs PCA reconstructions")
    fig.tight_layout()
    fig.savefig(outdir / "reconstructions.png", dpi=130)
    plt.close(fig)


def plot_metric_vs_k(results: dict[str, list[dict[str, Any]]], key: str, ylabel: str,
                     title: str, filename: str, outdir: Path) -> None:
    """Plots 2 and 3: a metric (PSNR or variance retained) against k, per method."""
    markers = {"power": "o", "jacobi": "x"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, res in results.items():
        ax.plot([r["k"] for r in res], [r[key] for r in res],
                marker=markers[method], linestyle="-", label=method)
    ax.set_xlabel("number of components k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=130)
    plt.close(fig)


def measure_runtimes(image_path: str, methods: list[str], sizes: list[int], k_max: int) -> dict[str, list[float]]:
    """Solver runtime for each method at each image size."""
    times: dict[str, list[float]] = {m: [] for m in methods}
    for size in sizes:
        X_centered, _ = pca.center_data(load_image(image_path, size))
        C = pca.covariance_matrix(X_centered)
        for method in methods:
            times[method].append(solve(C, method, min(k_max, size))["seconds"])
    return times


def plot_runtime(sizes: list[int], times: dict[str, list[float]], outdir: Path) -> None:
    """Plot 4: runtime against image size."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, values in times.items():
        ax.plot(sizes, values, marker="o", label=method)
    ax.set_xlabel("image size N (N x N pixels)")
    ax.set_ylabel("solver runtime [s]")
    ax.set_title("Runtime vs image size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "runtime_vs_size.png", dpi=130)
    plt.close(fig)


def plot_power_convergence(C: np.ndarray, outdir: Path) -> None:
    """Plot 5: Rayleigh quotient per iteration for the dominant eigenpair.

    Left: the quotient itself.  Right: its distance to the final value on a
    log scale; the slope reflects the rate (lambda_2 / lambda_1)^2.
    """
    history: list[float] = []
    eigen.power_method(C, history=history)
    values = np.array(history)
    error = np.abs(values - values[-1])
    iterations = np.arange(1, len(values) + 1)
    nonzero = error > 0  # the last point has error exactly 0 (log undefined)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(iterations, values)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("Rayleigh quotient")
    ax1.set_title("Power method: Rayleigh quotient")
    ax2.semilogy(iterations[nonzero], error[nonzero])
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("|quotient - final value|")
    ax2.set_title("Convergence (log scale)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "power_convergence.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def run_selftest() -> bool:
    """Check both solvers on a matrix whose eigenvalues are known by hand.

    A = [[ 2,-1, 0],
         [-1, 2,-1],
         [ 0,-1, 2]]
    det(A - lambda I) = (2 - lambda) * ((2 - lambda)^2 - 2) = 0
        =>  lambda = 2 + sqrt(2),  2,  2 - sqrt(2).
    """
    A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    expected = np.array([2.0 + np.sqrt(2.0), 2.0, 2.0 - np.sqrt(2.0)])

    power_vals, power_vecs, _ = eigen.power_method_with_deflation(A, 3, tol=1e-12)
    jacobi_vals, jacobi_vecs, _ = eigen.jacobi_eigen(A)

    all_ok = True
    # Power method: the Rayleigh quotient is accurate to ~tol, but the vector
    # converges only linearly, so its residual tolerance is looser.
    for name, vals, vecs, value_tol, resid_tol in [
            ("power", power_vals, power_vecs, 1e-8, 1e-4),
            ("jacobi", jacobi_vals, jacobi_vecs, 1e-9, 1e-9)]:
        value_ok = bool(np.allclose(vals, expected, atol=value_tol))
        # A v = lambda v  must hold for every returned pair
        residual = max(np.linalg.norm(A @ vecs[:, i] - vals[i] * vecs[:, i]) for i in range(3))
        vector_ok = bool(residual < resid_tol)
        print(f"[{'PASS' if value_ok and vector_ok else 'FAIL'}] {name:<6} "
              f"eigenvalues = {np.round(vals, 8)}  max residual |Av - lv| = {residual:.2e}")
        all_ok = all_ok and value_ok and vector_ok

    print(f"expected eigenvalues  = {np.round(expected, 8)}")
    print("Self-test", "PASSED" if all_ok else "FAILED")
    return all_ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return 0 if run_selftest() else 1

    outdir = BASE_DIR / "outputs"
    outdir.mkdir(exist_ok=True)
    methods = ["power", "jacobi"] if args.method == "both" else [args.method]

    # load -> center -> covariance
    X = load_image(args.image, args.size)
    ks = clean_k_values(args.k, args.size)
    X_centered, mean = pca.center_data(X)
    C = pca.covariance_matrix(X_centered)
    total_variance = float(np.trace(C))  # = sum of ALL eigenvalues
    k_max = max(ks)
    print(f"Image {args.image} resized to {args.size}x{args.size}; k = {ks}")

    # solve with each chosen method
    solutions = {m: solve(C, m, k_max) for m in methods}

    # validation table (console + CSV)
    rows = build_comparison_rows(C, solutions, k_max)
    print_comparison(rows, k_max)
    save_comparison_csv(rows, outdir / "comparison.csv")

    # compression metrics for each method and k
    results = {m: [evaluate_k(X, X_centered, mean, solutions[m], total_variance, k) for k in ks]
               for m in methods}
    for method in methods:
        print_metrics(method, results[method])

    # plots
    plot_reconstructions(X, results, outdir)
    plot_metric_vs_k(results, "psnr", "PSNR [dB]", "PSNR vs k", "psnr_vs_k.png", outdir)
    plot_metric_vs_k(results, "variance", "fraction of variance retained",
                     "Variance retained vs k", "variance_vs_k.png", outdir)
    sizes = [32, 48, 64, 96]
    plot_runtime(sizes, measure_runtimes(args.image, methods, sizes, k_max), outdir)
    plot_power_convergence(C, outdir)
    print(f"\nSaved comparison.csv and 5 plots to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
