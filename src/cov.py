import numpy as np


def gaussian(arr: np.array) -> np.ndarray:
    gamma = arr[:, np.newaxis] - arr[np.newaxis, :]
    return np.exp(-(gamma**2))


def laplacian(arr: np.array) -> np.ndarray:
    gamma = arr[:, np.newaxis] - arr[np.newaxis, :]
    return np.exp(-np.abs(gamma))
