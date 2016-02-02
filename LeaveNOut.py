__author__ = 'nhan'
import pdb;
import numpy as np

class LeaveNOut:
    def __init__(self):
        self.y_pred = []
        self.y_true = []

    def run(self, data = None, labels= None, model = None, n_out = 3):
        for i in range(len(data) / n_out):
            # Splitting data and labels into test and train parts
            test_data = data[n_out * i : n_out * (i + 1),:]
            train_data = np.vstack((data[: n_out *i, :], data[ n_out * (i + 1): ,:]))

            test_labels = labels[n_out * i : n_out * (i + 1), :]
            train_labels = np.vstack((labels[: n_out * i, :], labels[ n_out * (i + 1) : , :]))

            # Use train set to fit the model
            model.fit(train_data, train_labels)

            # Predict on the test set
            result  = model.predict(test_data)

            # Append predict and true labels to returned list
            for i in range(len(test_data)):
                self.y_pred.append(result[i])
                self.y_true.append(test_labels[i])

        return self.y_pred, self.y_true