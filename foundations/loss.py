import numpy as np
from numpy.typing import NDArray


class Solution:
    def binary_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        for i in range(y_pred.size):
            L += y_true[i] * np.log(y_pred[i] + 0.0000001) + (1 - y_true[i]) * np.log(
                1 - (y_pred[i] + 0.0000001)
            )
        L = (-1 / y_pred.size) * L
        return round(L, 4)

    def categorical_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        C = 0
        y_ic = 0
        for i in range(len(y_true)):
            for c in range(len(y_true[0])):
                L += y_true[i][c] * np.log(y_pred[i][c] + .0000001)
        
        return round(-L / len(y_true), 4)
