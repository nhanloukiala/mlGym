__author__ = 'nhan'
import pdb;
import numpy as np
from kNN import batch_l2distance

class LeaveNOut:
    def __init__(self, zone_radius = 0):
        self.zone_radius = zone_radius
        self.y_pred = []
        self.y_true = []

    def run(self, data=None, labels=None, model=None, n_out=3, coordinates = None):
        for i in range(len(data) / n_out):
            # Splitting data and labels into test and train parts
            test_data = data[n_out * i: n_out * (i + 1), :]
            train_data = np.vstack((data[: n_out * i, :], data[n_out * (i + 1):, :]))

            test_labels = labels[n_out * i: n_out * (i + 1), :]
            train_labels = np.vstack((labels[: n_out * i, :], labels[n_out * (i + 1):, :]))

            if self.zone_radius > 0 and (n_out > 1 or coordinates is None) :
                raise Exception("Dead zone radius is available only with leave 1 out and not-None coordinates metrics")

            if self.zone_radius > 0:
                test_coor = coordinates[n_out * i: n_out * (i + 1), :]
                train_coor = np.vstack((coordinates[: n_out * i, :], coordinates[n_out * (i + 1):, :]))
                train_labels, train_data = self.wipe_in_range(test_data, train_data, train_labels, test_coor, train_coor)

            # Use train set to fit the model
            model.fit(train_data, train_labels)

            # Predict on the test set
            result = model.predict(test_data)

            # Append predict and true labels to returned list
            for i in range(len(test_data)):
                self.y_pred.append(result[i])
                self.y_true.append(test_labels[i])

        return self.y_pred, self.y_true

    def wipe_in_range(self, test_data, train_data, train_labels, test_coor, train_coor):
        distances = batch_l2distance(np.tile(test_coor, (len(train_coor),1)), train_coor)
        indices = [index for index, x in enumerate(distances) if x > self.zone_radius]

        return train_labels[indices], train_data[indices]

