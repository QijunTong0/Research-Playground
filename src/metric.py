import numpy as np


def wasserstein(A, B) -> np.ndarray:
    return (
        np.linalg.trace(A)
        + np.linalg.trace(B)
        - 2 * np.sqrt(np.einsum("ij,ij->", A, B))
    )


def kr(A, B) -> np.ndarray:
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)
    return np.linalg.trace(A) + np.linalg.trace(B) - 2 * np.einsum("ij,ij", L.T, M)
