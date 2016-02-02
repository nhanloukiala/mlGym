__author__ = 'nhan'
import pandas as pd
from sklearn import preprocessing as prep
from config import *
import pandas as pd
from pandas import DataFrame
from KFold import KFold
from kNN import NearestNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from Score import *
import matplotlib.pyplot as plt
import seaborn as sns
from legacy_script import *
from LeaveNOut import *
from sklearn.svm import SVR
import pdb

data = pd.read_csv('Water_data.csv')
# Split data into labels and features
train_labels, train_data = np.hsplit(data, [3])

# Normalization
train_data = standardize_dataset(train_data)
train_labels = train_labels.as_matrix()

#Try with different neighbors and different leave N out cross validation:
for n in [1, 3]:
    plot_data = []

    for i in range(1, 30, 1):
        model = NearestNeighborsRegressor(n_neighbors=i)
        runner = LeaveNOut()
        predict, true = runner.run(data=train_data, model=model, labels=train_labels, n_out=n)
        score = Score()
        plot_data.append([score.c_score(np.array(predict)[:, 0], np.array(true)[:, 0]), i, 'c_total'])
        plot_data.append([score.c_score(np.array(predict)[:, 1], np.array(true)[:, 1]), i, 'Cd'])
        plot_data.append([score.c_score(np.array(predict)[:, 2], np.array(true)[:, 2]), i, 'Pb'])

    line_plot(np.array(plot_data), title="C_index by different K Neighbors - Leave %s out CV" % n,
              x_title="K Neighbors", y_title="C-Index")
