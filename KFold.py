__author__ = 'nhan'
import numpy as np
import random
import pdb


class KFold:
    def __init__(self, n, n_folds=2, shuffle=False):
        if not isinstance(n, int) or n is None:
            raise TypeError('n must be integer and not None')

        if n_folds > n:
            raise ArithmeticError('number of fold must not be ' +
                                  'larger than number of instances')

        self.n = n
        self.n_folds = n_folds
        self.shuffle = shuffle

    def get_indices(self):
        x = range(self.n)
        if self.shuffle:
            random.shuffle(x)

        fold_len = self.n / self.n_folds

        train = []
        test = []
        for i in range(self.n_folds):
            # index from i * n_folds -> (i + 1 ) * n_folds
            test.append(x[i * fold_len : (i + 1) * fold_len] if i < self.n_folds - 1 else x[i * fold_len : ])
            train.append((x[:i * fold_len] if i > 0 else []) +
                         (x[(i + 1) * fold_len : ] if i < self.n_folds - 1 else []))

        return train, test
