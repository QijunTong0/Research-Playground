import numpy as np
import ot


def wasserstein(A, B) -> np.ndarray:
    return ot.gaussian.bures_distance(A, B)


def kr(A, B) -> np.ndarray:
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)
    out = np.linalg.norm(np.einsum("ji,ji->i", L, M), ord=1)
    return np.sqrt(np.linalg.trace(A) + np.linalg.trace(B) - 2 * out)
