# Eigenvalue-Based Image Compression (PCA with Power and Jacobi Methods)

Compresses a grayscale image with PCA. The eigenvectors of the covariance
matrix come from two hand-written solvers (no `np.linalg.eig/eigh/svd` in
`eigen.py`):

- **Power method + deflation**: top-k eigenpairs, one at a time.
- **Cyclic Jacobi**: all eigenpairs via Givens rotations.

The N×N image is the data matrix X (N samples × N features). With centered
data X_c and covariance C = X_cᵀX_c/(N−1), keeping the top k eigenvectors V_k
gives Z = X_c V_k and X̂ = Z V_kᵀ + mean. Compression ratio = N² / (2Nk + N).
`np.linalg.eigh` is used only in `main.py` to validate the solvers.

## Files
`eigen.py` solvers · `pca.py` PCA + metrics · `main.py` CLI/experiments/plots ·
`images/sample.png` synthetic test image · `outputs/` results.

## Run
```
pip install -r requirements.txt
python main.py --selftest
python main.py --image images/sample.png --size 64 --k 5 10 20 40 --method both
```
Options: `--size` (default 64), `--k` (list), `--method power|jacobi|both`.
Outputs in `outputs/`: `comparison.csv`, `reconstructions.png`, `psnr_vs_k.png`,
`variance_vs_k.png`, `runtime_vs_size.png`, `power_convergence.png`.

## Sample results (64×64, default arguments)

| k | MSE | PSNR (dB) | compression ratio | variance retained |
|---|-----|-----------|-------------------|-------------------|
| 5 | 162.62 | 26.02 | 5.82 | 87.99 % |
| 10 | 99.46 | 28.15 | 3.05 | 92.65 % |
| 20 | 37.87 | 32.35 | 1.56 | 97.20 % |
| 40 | 0.70 | 49.69 | 0.79 | 99.95 % |

Both methods give identical metrics. Solver comparison (top 40 eigenpairs vs `eigh`):

| method | runtime (s) | count | max eigenvalue error | eigenvector error (max 1−\|v·u\|) |
|--------|-------------|-------|----------------------|-----------------------------------|
| power | 0.023 | 3318 iterations | 1.1e-4 | 2.8e-5 |
| jacobi | 0.165 | 11 sweeps | 5.6e-10 | ~0 |

Power is faster here because it computes only 40 of 64 pairs and each step is
one matrix-vector product, but it is less accurate: it converges slowly when
eigenvalues are close, and deflation accumulates error. Jacobi computes all
pairs to near machine precision. Runtimes vary by machine. Note that k = 40
exceeds the break-even point (ratio < 1), so it does not actually save space.
