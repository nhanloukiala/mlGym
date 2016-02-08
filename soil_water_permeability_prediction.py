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

input = pd.read_csv('./Soil_water_permeability_data/INPUT.csv',header=None)
output = pd.read_csv('./Soil_water_permeability_data/OUTPUT.csv', header=None)
coordinates = pd.read_csv('./Soil_water_permeability_data/COORDINATES.csv', header=None)

# Normalization
std_input = standardize_dataset(input)
plot_data = []

for n in range(0,201,10):
    model = NearestNeighborsRegressor(n_neighbors=5)
    runner = LeaveNOut(zone_radius=n)
    predict, true = runner.run(data=std_input, model=model, labels=output.as_matrix(), n_out=1, coordinates = coordinates.as_matrix())
    score = Score()
    plot_data.append([score.c_score(np.array(predict)[:, 0], np.array(true)[:, 0]), n, 'Concordance Index'])
    print "epoch %d " % n

line_plot(np.array(plot_data), title="Concordance index by different Dead zone radius - Leave 1 out CV",
              x_title="Dead zone radius", y_title="C-Index")


