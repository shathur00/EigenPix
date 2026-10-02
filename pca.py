"""PCA compression / reconstruction of an image and quality metrics.

Data model: an N x N grayscale image is treated as a data matrix X with
N samples (rows) and N features (columns).  PCA finds the directions
(eigenvectors of the covariance matrix) along which the rows vary most;
keeping only the top k directions gives a rank-k approximation.
"""

from __future__ import annotations

import numpy as np


def center_data(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the column means:  X_c = X - mean,  mean_j = (1/n) sum_i X_ij."""
    mean = X.mean(axis=0)
    return X - mean, mean


def covariance_matrix(X_centered: np.ndarray) -> np.ndarray:
    """Sample covariance  C = X_c^T X_c / (n - 1)  (computed by hand).

    C is d x d, symmetric and positive semi-definite, so all its eigenvalues
    are real and >= 0.  Eigenvalue lambda_i is the variance of the data along
    eigenvector v_i.
    """
    n = X_centered.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples to compute a covariance")
    return X_centered.T @ X_centered / (n - 1)


def _check_k(k: int, eigenvectors: np.ndarray) -> None:
    """Make sure 1 <= k <= number of available eigenvectors."""
    available = eigenvectors.shape[1]
    if not 1 <= k <= available:
        raise ValueError(f"k must satisfy 1 <= k <= {available}, got k={k}")


def compress(X_centered: np.ndarray, eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """Project centered data on the top-k eigenvectors:  Z = X_c V_k  (n x k).

    `eigenvectors` must have its columns sorted by descending eigenvalue.
    """
    _check_k(k, eigenvectors)
    return X_centered @ eigenvectors[:, :k]


def reconstruct(
    Z: np.ndarray, eigenvectors: np.ndarray, mean: np.ndarray, k: int
) -> np.ndarray:
    """Approximate the original data:  X_hat = Z V_k^T + mean.

    Because V_k has orthonormal columns, V_k V_k^T is the orthogonal projector
    onto the k-dimensional subspace, so X_hat is the best rank-k
    approximation (in least squares) of the data.
    """
    _check_k(k, eigenvectors)
    return Z @ eigenvectors[:, :k].T + mean


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mse(original: np.ndarray, approx: np.ndarray) -> float:
    """Mean squared error:  (1/(n d)) sum_ij (X_ij - X_hat_ij)^2."""
    return float(np.mean((original - approx) ** 2))


def psnr(original: np.ndarray, approx: np.ndarray, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB:  10 log10(peak^2 / MSE).

    Higher is better.  Perfect reconstruction (MSE = 0) gives +infinity.
    """
    error = mse(original, approx)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak**2 / error))


def compression_ratio(n_rows: int, n_cols: int, k: int) -> float:
    """Original size divided by compressed size, counted in stored numbers.

    Original : n_rows * n_cols values.
    Compressed must store
        Z    : n_rows * k   (the projected data)
        V_k  : n_cols * k   (the k eigenvectors, needed to decode)
        mean : n_cols       (column means, needed to un-center)

    ratio = (n_rows * n_cols) / (n_rows*k + n_cols*k + n_cols)

    A ratio > 1 means real compression; it falls below 1 for large k.
    """
    stored = n_rows * k + n_cols * k + n_cols
    return (n_rows * n_cols) / stored


def variance_retained(
    eigenvalues: np.ndarray, k: int, total_variance: float | None = None
) -> float:
    """Fraction of total variance kept by the top k components.

    retained = (lambda_1 + ... + lambda_k) / (lambda_1 + ... + lambda_d)

    The denominator equals trace(C).  When only the top eigenvalues were
    computed (power method), pass total_variance = trace(C) explicitly.
    `eigenvalues` must be sorted in descending order.
    """
    if not 1 <= k <= len(eigenvalues):
        raise ValueError(f"k must satisfy 1 <= k <= {len(eigenvalues)}, got k={k}")
    total = float(np.sum(eigenvalues)) if total_variance is None else total_variance
    return float(np.sum(eigenvalues[:k]) / total)
