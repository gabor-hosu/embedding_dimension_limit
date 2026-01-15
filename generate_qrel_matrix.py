import numpy as np


def disjoint_matrix(num_of_queries, k: int = 2) -> np.ndarray:
    A = np.zeros((num_of_queries, k * num_of_queries), dtype=bool)

    rows = np.arange(num_of_queries).reshape(num_of_queries, 1)  # (N,1)
    starts = k * np.arange(num_of_queries).reshape(num_of_queries, 1)  # (N,1)
    offsets = np.arange(k).reshape(1, k)  # (1, k)

    cols = starts + offsets  # (N, k), via broadcasting

    A[rows, cols] = True
    return A


def cyclic_matrix(num_of_queries, k: int = 2) -> np.array:
    num_of_cols = (num_of_queries - 1) * (k - 1) + 1
    A = np.zeros((num_of_queries, num_of_cols), dtype=bool)

    rows = np.arange(num_of_queries).reshape(num_of_queries, 1)  # (N, 1)
    starts = (k - 1) * np.arange(num_of_queries).reshape(num_of_queries, 1)  # (N, 1)
    offsets = np.arange(k).reshape(1, k)  # (1, k)

    cols = (starts + offsets) % num_of_cols

    A[rows, cols] = True
    return A
