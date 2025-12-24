from typing import Callable

import numpy as np


def brownian(x: np.ndarray):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    out = np.minimum(s, t)
    return out


def brownian_bridge(x: np.ndarray):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    out = np.minimum(s, t) - s * t
    return out


def integrated_brownian(x: np.ndarray):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    min_st = np.minimum(s, t)
    max_st = np.maximum(s, t)
    out = (min_st**2 * (3 * max_st - min_st)) / 6
    return out


def gaussian(x: np.ndarray, theta=1):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    out = np.exp(-((s - t) ** 2) / theta)
    return out


def laplacian(x: np.ndarray, theta=1.0):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    out = np.exp(-np.abs(s - t) / theta)
    return out


def matern32(x: np.ndarray, theta=1.0):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    d = np.abs(s - t)
    sqrt3_d = np.sqrt(3) * d / theta
    out = (1 + sqrt3_d) * np.exp(-sqrt3_d)
    return out


def polynomial(x: np.ndarray, degree=2, c=0):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    out = (s * t + c) ** degree
    return out


def periodic(x: np.ndarray, period=1.0, theta=1.0):
    s, t = x[np.newaxis, :], x[:, np.newaxis]
    sine_part = np.sin(np.pi * np.abs(s - t) / period) ** 2
    out = np.exp(-2 * sine_part / theta**2)
    return out


def bridge_kernel(x: np.ndarray, kernel: Callable, y: float = 0):
    cov = kernel(x)
    b_mean = (cov[:, -1] / cov[-1, -1]) * y
    b_cov = cov - cov[-1:, :] * cov[:, -1:] / cov[-1, -1]
    return b_mean, b_cov
