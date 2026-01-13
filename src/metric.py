import numpy as np


def wasserstein(A, B) -> np.ndarray:
    return np.einsum("ij,ij->", A, B)


def kr(A, B) -> np.ndarray:
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(A)
    return np.einsum("ij,ij", L.T, M)
