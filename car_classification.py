__author__ = 'nhan'
import pandas as pd
from sklearn import preprocessing as prep
from config import *
import pandas as pd
from pandas import DataFrame
from KFold import KFold
from kNN import NearestNeighborsClassifier
from Score import *
import matplotlib.pyplot as plt
import seaborn as sns
from legacy_script import *

data = pd.read_csv('cardata.csv')

########Encode from categorical to numerical data
encoder = prep.LabelEncoder()
ndata = DataFrame()
for col in data.columns:
    encoder.fit(uniques[col])
    ndata[col] = encoder.transform(data[col])

# Simple Explorartory Analysis
# Bar plots
for col in data.columns:
    agg = data.groupby([col, 'class']).count()
    sns.set_style("whitegrid")
    ax = sns.barplot(x=agg.ix[:, [0]].index.values, y=agg.ix[:, [0]].values.T[0])
    plt.title("Distribution of " + col)
    plt.show()

# PCA
pca_cal(standardize_dataset(ndata.as_matrix()), data['class'], data.columns, title="PCA with normalization")

# Seperate dataset to test and train set
kf = KFold(n=len(ndata), n_folds=10, shuffle=True)
train, test = kf.get_indices()
s = Score()

total_train_error = []
total_test_error = []

for k in range(1, 200, 3):
    train_error = []
    test_error = []
    for i in range(10):
        print "round %d %d" % (k, i)
        train_data = ndata.ix[train[i]]
        test_data = ndata.ix[test[i]]

        #######TRAIN#####
        nn = NearestNeighborsClassifier(n_neighbors=k)

        # convert from panda frame to numpy matrix
        nn.fit(train_data[features].as_matrix(), train_data[target_feature].as_matrix())

        train_predicted = nn.predict(train_data[features].as_matrix())
        test_predicted = nn.predict(test_data[features].as_matrix())

        # Scoring
        test_mis_rate = s.misclassification_rate(np.array(test_predicted), test_data[target_feature].values.T[0])
        train_mis_rate = s.misclassification_rate(np.array(train_predicted), train_data[target_feature].values.T[0])
        test_error.append(test_mis_rate)
        train_error.append(train_mis_rate)

    total_train_error.append([np.mean(np.abs(train_error)), k, 1])
    total_test_error.append([np.mean(np.abs(test_error)), k, 0])

np_stage = np.vstack((np.array(total_train_error), np.array(total_test_error)))

plot_data = DataFrame()
plot_data['x'] = np_stage[:, 1].astype(int)
plot_data['y'] = np_stage[:, 0]

plot_data['Train=1/Test=0'] = np_stage[:, 2].astype(int)
# plot_data = pd.read_csv('plot.csv')
plot_data.to_csv('plot.csv')

sns.set(style="whitegrid")
g = sns.pointplot(x="x", y="y", hue="Train=1/Test=0", data=plot_data, label='Train = 1.0 / Test = 0.0')
plt.title("K Neighbors vs  Misclassification Rate among Train/Test sets", fontsize=25)
plt.ylabel("Misclassification Rate", fontsize=12)
plt.xlabel("K Neighbors", fontsize=12)
plt.show()
