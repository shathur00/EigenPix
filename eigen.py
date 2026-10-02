"""Custom eigenvalue solvers for real symmetric matrices.

Two classical algorithms are implemented from scratch:

* Power method (with deflation) -- finds the largest eigenpairs one at a time.
* Cyclic Jacobi method          -- finds ALL eigenpairs at once by rotating
                                   the matrix towards diagonal form.

Only basic numpy array operations (matmul, outer, norm, ...) are used here.
np.linalg.eig / eigh / svd are deliberately NOT used anywhere in this file.
"""

from __future__ import annotations

import warnings

import numpy as np


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------
def _as_square_matrix(A: np.ndarray) -> np.ndarray:
    """Return A as a float 2-D array, raising ValueError unless it is square."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    return A


def _random_unit_vector(n: int, seed: int) -> np.ndarray:
    """Random start vector x0 with ||x0|| = 1 (fixed seed => reproducible)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    length = np.linalg.norm(x)
    if length == 0.0:
        raise ValueError("random start vector has zero length")
    return x / length


# --------------------------------------------------------------------------
# Power method
# --------------------------------------------------------------------------
def power_method(
    A: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 10000,
    seed: int = 0,
    history: list[float] | None = None,
) -> tuple[float, np.ndarray, int]:
    """Dominant eigenpair of A by normalized power iteration.

    Math
    ----
    Starting from a random unit vector x_0, repeat

        y_{i+1} = A x_i              (multiply)
        x_{i+1} = y_{i+1} / ||y_{i+1}||   (normalize, avoids overflow)
        lambda_{i+1} = x_{i+1}^T A x_{i+1}   (Rayleigh quotient)

    Writing x_0 in the eigenbasis, x_0 = sum_j c_j v_j, gives
    A^i x_0 = sum_j c_j lambda_j^i v_j.  The term with the largest |lambda|
    dominates, so x_i -> v_1 at rate |lambda_2 / lambda_1|^i.
    The Rayleigh quotient is exact for an eigenvector and its error is
    quadratic in the vector error, so it converges faster than x_i itself.

    We stop when the relative change of the Rayleigh quotient is below `tol`.

    Edge case: if A x = 0 the vector lies in the null space of A, i.e. it is
    an eigenvector for eigenvalue 0.  We return (0.0, x, iteration) instead
    of dividing by zero.  This happens e.g. when deflation has removed every
    non-zero eigenvalue.

    Parameters
    ----------
    A : (n, n) matrix.  Convergence needs a unique dominant |eigenvalue|.
    history : optional list; if given, every Rayleigh quotient is appended
              to it (used for the convergence plot).

    Returns
    -------
    (eigenvalue, unit eigenvector, number of iterations)
    """
    A = _as_square_matrix(A)
    x = _random_unit_vector(A.shape[0], seed)
    # "Zero" threshold relative to the size of A (Frobenius norm).
    zero_tol = 1e-14 * np.linalg.norm(A)
    smallest = np.finfo(float).tiny  # avoids dividing by 0 in the relative test

    eigenvalue = 0.0
    for iteration in range(1, max_iter + 1):
        y = A @ x
        y_norm = np.linalg.norm(y)
        if y_norm <= zero_tol:  # A x = 0  ->  eigenvalue 0
            return 0.0, x, iteration
        x = y / y_norm
        new_value = float(x @ A @ x)  # Rayleigh quotient (x has length 1)
        if history is not None:
            history.append(new_value)
        change = abs(new_value - eigenvalue)
        if iteration > 1 and change <= tol * max(abs(new_value), smallest):
            return new_value, x, iteration
        eigenvalue = new_value

    warnings.warn("power_method did not converge within max_iter", stacklevel=2)
    return eigenvalue, x, max_iter


def power_method_with_deflation(
    A: np.ndarray,
    k: int,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k eigenpairs of a symmetric matrix via power method + deflation.

    Math
    ----
    If (lambda_1, v_1) is an eigenpair of a symmetric A (||v_1|| = 1), then

        A' = A - lambda_1 v_1 v_1^T

    has the same eigenpairs as A except that lambda_1 is replaced by 0
    (because v_1^T v_j = 0 for j != 1).  The power method applied to A'
    therefore finds the NEXT largest eigenpair.  Repeat k times.

    Caveat: v_1 is only known approximately, so small errors accumulate in
    later components.

    Returns
    -------
    eigenvalues : (k,) in the order found (descending |lambda|)
    eigenvectors: (n, k), column i belongs to eigenvalues[i]
    iterations  : (k,) number of power iterations used by each component
    """
    A = _as_square_matrix(A)
    n = A.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}, got k={k}")

    deflated = A.copy()  # never modify the caller's matrix
    eigenvalues = np.zeros(k)
    eigenvectors = np.zeros((n, k))
    iterations = np.zeros(k, dtype=int)

    for i in range(k):
        value, vector, count = power_method(deflated, tol, max_iter)
        eigenvalues[i] = value
        eigenvectors[:, i] = vector
        iterations[i] = count
        deflated = deflated - value * np.outer(vector, vector)  # A <- A - lambda v v^T
    return eigenvalues, eigenvectors, iterations


# --------------------------------------------------------------------------
# Jacobi method
# --------------------------------------------------------------------------
def _off_diagonal_norm(D: np.ndarray) -> float:
    """Frobenius norm of the off-diagonal part: sqrt(sum_{i != j} D_ij^2)."""
    off = D - np.diag(np.diag(D))
    return float(np.linalg.norm(off))


def _jacobi_rotate(D: np.ndarray, V: np.ndarray, p: int, q: int) -> None:
    """Zero D[p, q] with one Givens rotation, updating D and V IN PLACE.

    Math
    ----
    Let J be the identity except for the 2x2 block in rows/cols (p, q):

            J[p,p] =  c   J[p,q] = s
            J[q,p] = -s   J[q,q] = c

    Then D' = J^T D J is similar to D (same eigenvalues).  Choosing

        theta = (D_qq - D_pp) / (2 D_pq)
        t     = sign(theta) / (|theta| + sqrt(theta^2 + 1))   (smaller root of
                                                              t^2 + 2 theta t - 1 = 0)
        c     = 1 / sqrt(t^2 + 1),   s = t c

    makes D'_pq = 0.  Multiplying by J only changes rows p, q (from the left)
    and columns p, q (from the right), so we update just those -- O(n) work
    instead of building the full n x n matrix J (O(n^3)).

    The eigenvector matrix accumulates the product of rotations:
    V <- V J  (so at the end V = J_1 J_2 ... J_m and D = V^T A V is diagonal).
    """
    d_pq = D[p, q]
    if d_pq == 0.0:
        return  # already zero, nothing to rotate

    theta = (D[q, q] - D[p, p]) / (2.0 * d_pq)
    sign = 1.0 if theta >= 0.0 else -1.0
    t = sign / (abs(theta) + np.hypot(theta, 1.0))  # hypot = sqrt(theta^2+1)
    c = 1.0 / np.hypot(t, 1.0)
    s = t * c

    # D <- J^T D : mix rows p and q
    row_p, row_q = D[p, :].copy(), D[q, :].copy()
    D[p, :] = c * row_p - s * row_q
    D[q, :] = s * row_p + c * row_q
    # D <- D J : mix columns p and q
    col_p, col_q = D[:, p].copy(), D[:, q].copy()
    D[:, p] = c * col_p - s * col_q
    D[:, q] = s * col_p + c * col_q
    # Remove rounding residue: the rotation was built to make these exactly 0.
    D[p, q] = 0.0
    D[q, p] = 0.0

    # V <- V J : accumulate the rotation into the eigenvector matrix
    v_p, v_q = V[:, p].copy(), V[:, q].copy()
    V[:, p] = c * v_p - s * v_q
    V[:, q] = s * v_p + c * v_q


def jacobi_eigen(
    A: np.ndarray,
    tol: float = 1e-10,
    max_sweeps: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """All eigenpairs of a real symmetric matrix by the cyclic Jacobi method.

    Math
    ----
    One *sweep* rotates every pair (p, q), p < q, once (in row order).  Each
    rotation zeroes one off-diagonal entry (it may slightly refill others,
    but the total off-diagonal energy

        off(D)^2 = sum_{i != j} D_ij^2

    strictly decreases, and the convergence is eventually quadratic).  When
    off(D) is tiny, D ~ diag(lambda_1..lambda_n) and the accumulated V holds
    the eigenvectors:  V^T A V = D.

    We stop when off(D) <= tol * ||A||_F (a relative test).

    Raises
    ------
    ValueError if A is not square or not symmetric.

    Returns
    -------
    eigenvalues (descending), eigenvectors (columns, same order), sweeps used
    """
    A = _as_square_matrix(A)
    scale = np.linalg.norm(A)
    if not np.allclose(A, A.T, rtol=1e-8, atol=1e-12 * max(scale, 1.0)):
        raise ValueError("jacobi_eigen requires a symmetric matrix (A != A^T)")

    n = A.shape[0]
    D = A.copy()
    V = np.eye(n)
    threshold = tol * scale

    sweeps = 0
    while sweeps < max_sweeps and _off_diagonal_norm(D) > threshold:
        for p in range(n - 1):
            for q in range(p + 1, n):
                _jacobi_rotate(D, V, p, q)
        sweeps += 1

    if _off_diagonal_norm(D) > threshold:
        warnings.warn("jacobi_eigen did not converge within max_sweeps", stacklevel=2)

    order = np.argsort(np.diag(D))[::-1]  # largest eigenvalue first
    return np.diag(D)[order], V[:, order], sweeps
