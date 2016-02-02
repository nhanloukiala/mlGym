__author__ = 'nhan'

import numpy as np
import operator
import pdb


def l2distance(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError('Shape of vectors does not match')

    return np.sum((v1 - v2) ** 2) ** 0.5


def batch_l2distance(mat1, mat2):
    if mat1.shape != mat2.shape:
        raise ValueError('Shape of vectors does not match')

    return np.sum((mat1 - mat2) ** 2, axis=1) ** 0.5


class NearestNeighbors:
    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        results = []
        # Iterate through the input matrix and compute result for each row
        for i in range(x.shape[0]):
            top_labels, top_dists = self.getTopK(x[i, :])
            results.append(self.elect(top_dists=top_dists, labels=top_labels))
        return results

    def elect(self, top_dists, labels, axis=0):
        pass

    def getTopK(self, x):
        # Calculate distance vector
        dist_vec = batch_l2distance(self.X, np.tile(x, (self.X.shape[0], 1)))

        # Sort and get k labels from indices
        indices = np.argsort(dist_vec)[:self.n_neighbors]
        top_labels = self.y[indices, :]
        top_dists = dist_vec[indices]

        return top_labels, top_dists

class NearestNeighborsRegressor(NearestNeighbors):
    def elect(self, top_dists, labels, axis=0):
        return np.mean(labels, axis=0)


class NearestNeighborsClassifier(NearestNeighbors):
    def elect(self, top_dists, labels, axis=0):
        # Count Frequency and Sum Distance of each label.
        freq = {}
        for i in range(len(labels)):
            if labels[i, 0] not in freq:
                freq[labels[i, 0]] = [1, top_dists[i]]
            else:
                freq[labels[i, 0]][0] += 1
                freq[labels[i, 0]][1] += top_dists[i]

        # 1/ Sort by Frequency
        # 2/ If 2 labels have same frequency, favor the one with lower Sum Distance
        results = sorted(freq.items(), key=lambda x: (x[1][0], -x[1][1]), reverse=True)
        return results[0][0]

