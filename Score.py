__author__ = 'nhan'
from numpy import nditer
import numpy as np


class Score:
    def __init__(self):
        pass

    # numpy array
    def misclassification_rate(self, y_pred, y_true):
        if y_pred.ndim < 2:
            np.expand_dims(y_pred, axis=1)

        # import pdb;pdb.set_trace()

        mis_rate = 0.0

        for x, y in nditer([y_pred, y_true], ['refs_ok']):
            mis_rate += 1.0 if x != y else 0.0

        return mis_rate / len(y_pred)

    def accuracy(self, y_pred, y_true):
        return 1 - self.misclassification_rate(y_pred, y_true)

    def c_score(self, y_pred, y_true):
        n = 0.0
        h_num = 0.0
        for i in range(len(y_true)):
            t = y_true[i]
            p = y_pred[i]

            for j in range(i+1, len(y_true)):
                nt = y_true[j]
                np = y_pred[i]

                if t != nt:
                    n += 1
                    if (p < np and t < nt) or (p > np and t > nt):
                        h_num += 1
                    elif (p < np and t > nt) or (p > np and t < nt):
                        pass
                    else:
                        h_num += 0.5

        return h_num / n

    def cost_matrix(self, y_pred, y_true, n_labels):
        y_pred_num = y_pred
        y_true_num = y_true

        # fill in the table index-wise
        table = np.zeros((n_labels, n_labels), dtype=np.int)
        for y_t, y_p in nditer([y_true_num, y_pred_num]):
            table[y_t, y_p] += 1

        return table
