import numpy as np
import ot
from scipy.linalg import solve_triangular


def wasserstein(A, B) -> np.ndarray:
    return ot.gaussian.bures_distance(A, B)


def kr(A, B) -> np.ndarray:
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)
    out = np.linalg.norm(np.einsum("ji,ji->i", L, M), ord=1)
    return np.sqrt(np.linalg.trace(A) + np.linalg.trace(B) - 2 * out)


def kl_divergence(Sigma0, Sigma1):
    """
    Computes KL(N(0, Sigma0) || N(0, Sigma1)) optimized for CPU/Numpy.

    Args:
        Sigma0: Covariance matrix of P (d, d) - numpy array
        Sigma1: Covariance matrix of Q (d, d) - numpy array

    Returns:
        float: The KL divergence value
    """
    # 1. Compute Cholesky Decompositions
    # lower=True returns lower triangular matrix L such that Sigma = L @ L.T
    # This is O(d^3) but highly optimized in LAPACK
    L0 = np.linalg.cholesky(Sigma0)
    L1 = np.linalg.cholesky(Sigma1)

    # 2. Compute Log-Determinant Term
    # log(|Sigma|) = 2 * sum(log(diag(L)))
    # We need: log|Sigma1| - log|Sigma0|
    log_det_1 = 2 * np.sum(np.log(np.diag(L1)))
    log_det_0 = 2 * np.sum(np.log(np.diag(L0)))
    log_det_term = log_det_1 - log_det_0

    # 3. Compute Trace Term: tr(Sigma1^{-1} @ Sigma0)
    # Equivalent to || L1^{-1} @ L0 ||_F^2
    # We solve the system: L1 @ X = L0 for X
    # solve_triangular is much faster than np.linalg.solve
    X = solve_triangular(L1, L0, lower=True)

    # The trace of (X @ X.T) is simply the sum of all elements squared (Frobenius norm squared)
    trace_term = np.sum(X**2)

    # 4. Combine
    d = Sigma0.shape[0]
    kl = 0.5 * (trace_term - d + log_det_term)

    return kl
