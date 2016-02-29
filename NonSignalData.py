__author__ = 'nhan'
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from LeaveNOut import *
from Score import Score
from sklearn.metrics import auc
import pandas as pd
from scipy.stats import kendalltau
import operator
import seaborn as sns
import matplotlib.pyplot as plt
from legacy_script import *

mu, sigma = 0, 0.1 # mean and standard deviation
ssize = [10, 50, 100, 500]




def firstProblem():
    result = []

    for m in range(100):
        for size in ssize:
            score = Score()
            cv = LeaveNOut()
            classifier = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree")

            X = np.random.normal(mu, sigma, size)
            X = np.expand_dims(X, axis=1)

            Y = np.zeros([X.shape[0]])
            Y = np.expand_dims(Y, axis=1)
            one_indices = random.sample([i for i in range(X.shape[0])], size / 2)
            Y[one_indices, 0] = 1

            #leave one out
            pred, true = cv.run(data=X, labels=Y, model=classifier, n_out=1)
            c = score.c_score(pred, true)
            auc_score = auc(pred, true, reorder=True)
            result.append([c, auc_score, size, m])

    data = pd.DataFrame(result)
    data.to_csv('result1.csv', header=False, index=False)




# re = select(X, Y, select_count=100)
def wrong_feature_selection(X, Y):
    score = Score()
    cv = LeaveNOut()
    classifier = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree")

    X, indices = cv.select(X, Y, select_count=10)
    pred, true = cv.run(data=X, labels=Y, model=classifier, n_out=1)
    c = score.c_score(pred, true)
    auc_score = auc(pred, true, reorder=True)
    return c, auc_score

def right_feature_selection(X, Y):
    score = Score()
    cv = LeaveNOut()
    classifier = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree")

    pred, true = cv.run(data=X, labels=Y, model=classifier, n_out=1, embedded_feature_selection=True)
    c = score.c_score(pred, true)
    auc_score = auc(pred, true, reorder=True)
    return c, auc_score


#second problem
X = np.random.normal(mu, sigma, [50,1000])
Y = np.zeros([X.shape[0]])
Y = np.expand_dims(Y, axis=1)
one_indices = random.sample([i for i in range(X.shape[0])], X.shape[0] / 2)
Y[one_indices, 0] = 1

c_score, auc = wrong_feature_selection(X, Y)
print "Feature selection done wrong c_score : " + c_score
print "Feature selection done wrong auc : " + auc

c_score, auc = right_feature_selection(X, Y)
print "Feature selection done right c_score : " + c_score
print "Feature selection done right auc : " + auc

# First problem
firstProblem()
data = pd.read_csv('result1.csv')

line_plot(data[['c_score', 'iter',  'sample_size']].as_matrix(), 'C_score score among different iteration', 'iter','c_score', legend_label='Sample Size')
line_plot(data[['auc', 'iter',  'sample_size']].as_matrix(), 'AUC score among different iteration', 'iter','auc', legend_label='Sample Size')
