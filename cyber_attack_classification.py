__author__ = 'nhan'
import pandas as pd
from sklearn import preprocessing as prep
from config import *
from sklearn.pipeline import *
from sklearn.svm import *
from sklearn.ensemble import *
import pandas as pd
from pandas import DataFrame
from KFold import KFold
# from kNN import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import *
from Score import *
import matplotlib.pyplot as plt
import seaborn as sns
from legacy_script import *
import numpy as np
from KFold import KFold
from sklearn.metrics import *
import pdb

def line_plot(data ,title = "", x_title ="", y_title="", legend_label="",group_labels=None):
    plot_data = DataFrame()

    plot_data['x'] = data[:, 1].astype(int)
    plot_data['y'] = data[:, 0].astype(float)

    plot_data[legend_label] = data[:, 2]
    sns.set(style="whitegrid")
    g = sns.pointplot(x="x", y="y", hue=legend_label, data=plot_data, hue_order=np.unique(plot_data[legend_label]))
    plt.title(title, fontsize=25)
    plt.ylabel(y_title, fontsize=12)
    plt.xlabel(x_title, fontsize=12)
    plt.show()

df = pd.read_csv('./KDD_CUP99/TrainData.csv')
df_test = pd.read_csv('./KDD_CUP99/TestData.csv')
df_test = df_test.dropna()

matrix_test = df_test.as_matrix()
matrix = df.as_matrix()
data, labels = np.hsplit(matrix, [matrix.shape[1] - 1])
test_data, test_labels = np.hsplit(matrix_test, [matrix_test.shape[1] - 1])
test_labels = test_labels.T[0].astype(int).tolist()
general_matrix = np.vstack((data, test_data))

def Encoding(data, general_matrix=None):
    encoder = LabelBinarizer()
    count = 0
    # encoding
    for i in range(data.shape[1]):
        if type(data[0, i]) == str:
            count += 1
            col = data[:, i]
            unique = np.unique(col if general_matrix is None else general_matrix[:, i])

            try:
                encoder.fit(unique)
            except:
                pass

            new_col = encoder.transform(col)

            # split at i and i + 1
            before, removed, after = np.hsplit(data, [i, i + 1])
            # concatenate
            data = np.concatenate((before, new_col, after), axis=1)
            before, removed, after = np.hsplit(general_matrix, [i, i + 1])
            general_matrix = np.concatenate((before, encoder.transform(general_matrix[:, i]), after), axis=1)

    print "count : %d" % count
    # return data
    return data


data = Encoding(data, general_matrix)
test_data = Encoding(test_data, general_matrix)

p = pca(n_components=50)
data[:, 4:] = standardize_dataset(data[:, 4:])
test_data[:, 4:] = standardize_dataset(test_data[:, 4:])

# Seperate dataset to test and train set
kf = KFold(n=len(data), n_folds=10, shuffle=True)
train, test = kf.get_indices()
s = Score()

total_cv_error = []
total_test_error = []
confusion_matx = []
f1score = []
for k in range(1, 15, 3):
    cv_error = []
    test_error = []
    nn = Pipeline([
            # ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
            ('classification', KNeighborsClassifier(n_neighbors=k, metric='manhattan'))
        ])
    for i in range(10):
        print "round %d %d" % (k, i)
        train_data = data[train[i]]
        train_labels = labels[[train[i]]]
        cv_data = data[test[i]]
        cv_labels = labels[test[i]]

        #######TRAIN#####
        # nn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

        # convert from panda frame to numpy matrix
        nn.fit(train_data, train_labels.T[0].tolist())
        # Scoring
        cv_predicted = nn.predict(cv_data)
        cv_mis_rate = s.misclassification_rate(np.array(cv_predicted, dtype=int), cv_labels.T[0])
        cv_error.append(cv_mis_rate)

    #Test set
    test_predicted = nn.predict(test_data).tolist()
    test_mis_rate = s.misclassification_rate(np.array(test_predicted, dtype=int), test_labels)
    test_error.append(test_mis_rate)
    #confusion matrix
    if k == 1 :
        print confusion_matrix(y_true= test_labels, y_pred=test_predicted, labels = np.unique(np.concatenate((test_labels, test_predicted), axis=1)).tolist())

    #F1 score test set
    f1score.append([f1_score(test_labels, test_predicted, average='micro'), k , 'Micro'])
    f1score.append([f1_score(test_labels, test_predicted, average='macro'), k , 'Macro'])

    total_cv_error.append([np.mean(np.abs(cv_error)), k, 'Train'])
    total_test_error.append([np.mean(np.abs(test_error)), k, 'Test'])

np_stage = np.vstack((np.array(total_cv_error), np.array(total_test_error)))
f1score = np.array(f1score)
line_plot(data=np_stage ,title = "K Neighbors vs  Misclassification Rate among Train/Test sets", x_title ="K Neighbors", y_title="Misclassification Rate", legend_label="Test / Train set",group_labels=np.unique(np_stage[:, 2]))
line_plot(data=f1score ,title = "K Neighbors vs  F1 score on test set", x_title ="K Neighbors", y_title="F1 score Rate", legend_label="Macro /  Micro",group_labels=f1score[:, 2])


