import numpy as np


def gaussian(arr: np.ndarray) -> np.ndarray:
    gamma = arr[:, np.newaxis] - arr[np.newaxis, :]
    return np.exp(-(gamma**2))


def laplacian(arr: np.ndarray) -> np.ndarray:
    gamma = arr[:, np.newaxis] - arr[np.newaxis, :]
    return np.exp(-np.abs(gamma))


def generate(
    lim_l=0, lim_r=50, num=100, shuffle: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(lim_l, lim_r, num)
    if shuffle:
        t = np.array(list(t)[::2] + list(t)[1::2])
    C_m = gaussian(t)
    C_n = laplacian(t)
    return C_m, C_n
