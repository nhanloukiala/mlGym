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
train_labels, train_data = np.hsplit(data, [3])
train_data = standardize_dataset(train_data)
train_labels = train_labels.as_matrix()

plot_data = []

for i in range(1, 30, 1):
    model = NearestNeighborsRegressor(n_neighbors = i)
    runner = LeaveNOut()
    predict, true = runner.run(data=train_data, model=model, labels=train_labels, n_out=3)
    score = Score()

    plot_data.append([score.c_score(np.array(predict)[:,0], np.array(true)[:,0]), i, 'c_total'])
    plot_data.append([score.c_score(np.array(predict)[:,1], np.array(true)[:,1]), i, 'Cd'])
    plot_data.append([score.c_score(np.array(predict)[:,2], np.array(true)[:,2]), i, 'Pb'])

line_plot(np.array(plot_data))