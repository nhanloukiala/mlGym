__author__ = 'nhan'
from numpy import nditer
import numpy as np


class Score:
    def __init__(self):
        pass

    # numpy array
    def misclassification_rate(self, y_pred, y_true):
        mis_rate = 0.0
        for x, y in nditer([y_pred, y_true]):
            mis_rate += 1.0 if x != y else 0.0

        return mis_rate / len(y_pred)

    def cost_matrix(self, y_pred, y_true, n_labels):
        y_pred_num = y_pred
        y_true_num = y_true

        # fill in the table index-wise
        table = np.zeros((n_labels, n_labels), dtype=np.int)
        for y_t, y_p in nditer([y_true_num, y_pred_num]):
            table[y_t, y_p] += 1

        return table
