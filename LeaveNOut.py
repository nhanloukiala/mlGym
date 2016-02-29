__author__ = 'nhan'
import pdb;
import numpy as np
from kNN import batch_l2distance
from scipy.stats import kendalltau
import operator
import math


class LeaveNOut:
    def __init__(self, zone_radius=0, symmetric_cross_validation = False):
        self.zone_radius = zone_radius
        self.y_pred = []
        self.y_true = []
        self.symmertric = symmetric_cross_validation

    def run(self, data=None, labels=None, model=None, n_out=3, coordinates=None, embedded_feature_selection=False):
        for i in range(len(data) / n_out):
            # Splitting data and labels into test and train parts
            test_data = data[n_out * i: n_out * (i + 1), :]
            train_data = np.vstack((data[: n_out * i, :], data[n_out * (i + 1):, :]))

            test_labels = labels[n_out * i: n_out * (i + 1), :]
            train_labels = np.vstack((labels[: n_out * i, :], labels[n_out * (i + 1):, :]))

            # Apply optional Dead-zone radius
            train_data, train_labels = self.applyOptionalDeadzoneRadius(coordinates, i, n_out, test_data, train_data,
                                                                        train_labels)

            #Apply embedded feature selections
            if embedded_feature_selection:
                train_data, indices = self.select(train_data, train_labels)
                test_data = test_data[:,indices]

            # Apply optional Symmetric Pair Cross-Validation
            if self.symmertric == True:
                square_length = math.sqrt(len(data))
                row = i / square_length
                col = i % square_length

                if(n_out != 1 or (len(data) % square_length) != 0):
                    raise Exception("Invalid parameter, symmetric only works with leave 1 out and square-shaped data set")

                train_data = np.delete(np.array(data, copy=True), [x for x in range(len(data)) if (x / square_length == row or x % square_length == row or x % square_length == col or x / square_length == col)], axis=0)
                train_labels = np.delete(np.array(labels, copy=True), [x for x in range(len(data)) if (x / square_length == row or x % square_length == row or x % square_length == col or x / square_length == col)], axis=0)

            # Use train set to fit the model
            model.fit(train_data, train_labels)

            # Predict on the test set
            result = model.predict(test_data)

            # Append predict and true labels to returned list
            for i in range(len(test_data)):
                self.y_pred.append(result[i])
                self.y_true.append(test_labels[i] if not type(test_labels[i]) == np.ndarray else test_labels[i][0])

        return self.y_pred, self.y_true

    def applyOptionalDeadzoneRadius(self, coordinates, i, n_out, test_data, train_data, train_labels):
        if self.zone_radius > 0 and (n_out > 1 or coordinates is None):
            raise Exception("Dead zone radius is available only with leave 1 out and not-None coordinates metrics")
        if self.zone_radius > 0:
            test_coor = coordinates[n_out * i: n_out * (i + 1), :]
            train_coor = np.vstack((coordinates[: n_out * i, :], coordinates[n_out * (i + 1):, :]))
            train_labels, train_data = self.wipe_in_range(test_data, train_data, train_labels, test_coor, train_coor)

        return train_data, train_labels

    def deleteRow(self, row_number, data):
        first, deleted, second = np.vsplit(data, [row_number, row_number + 1])

        return np.vstack((first, second))

    def wipe_in_range(self, test_data, train_data, train_labels, test_coor, train_coor):
        distances = batch_l2distance(np.tile(test_coor, (len(train_coor), 1)), train_coor)
        indices = [index for index, x in enumerate(distances) if x > self.zone_radius]

        return train_labels[indices], train_data[indices]

    def select(self, X, Y, select_count=100):
        corr = []

        for i in range(X.shape[1]):
            kd = kendalltau(X[:, i], Y)
            corr.append((i, abs(kd.correlation)))

        corr = sorted(corr, key=operator.itemgetter)[0:select_count]
        indices = [x for x, y in corr]
        return X[:, indices], indices