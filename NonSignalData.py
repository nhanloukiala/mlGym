__author__ = 'nhan'
import numpy as np
import pdb
import random
from sklearn.neighbors import KNeighborsClassifier
from LeaveNOut import *
from Score import Score
from sklearn.metrics import auc
import pandas as pd
from scipy.stats import kendalltau
mu, sigma = 0, 0.1 # mean and standard deviation
ssize = [10, 50, 100, 500]
score = Score()
classifier = KNeighborsClassifier(n_neighbors=3)
cv = LeaveNOut()

result = []

for i in range(100):
    for size in ssize:
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
        result.append([c, auc_score, size])

data = pd.DataFrame(result)
data.to_csv('result.csv', header=False, index=False)

# def select(X, Y, select_count):
#     for i in range(X.shape[1]):

