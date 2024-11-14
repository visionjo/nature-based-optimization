import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function, widely used for testing optimization algorithms."""
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    """Ackley function, characterized by a nearly flat outer region and a central peak."""
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq = np.sum(x ** 2) / len(x)
    cos_sum = np.sum(np.cos(c * x)) / len(x)
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(cos_sum) + a + np.e


def sphere(x: np.ndarray) -> float:
    """Sphere function, simple and unimodal, often used as a baseline for optimization."""
    return np.sum(x ** 2)
