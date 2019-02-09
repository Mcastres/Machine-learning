import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import sys
import math
from sklearn import datasets, linear_model

# Read the csv dataset and return an array of it
def read_csv_file(file):
    dataset = pd.read_csv(sys.argv[1])
    x = dataset.iloc[:, 6:].values
    y = dataset.iloc[:, 1].values
    y = np.unique(y, return_inverse=True)[1].tolist()
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(x)
    x = imputer.transform(x)
    return [x, y]

class LogisticRegression(object):

    def __init__(self, learning_rate=0.001, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w_ = []

        for i in np.unique(y):
            y_copy = [1 if c == i else 0 for c in y]
            w = np.ones(X.shape[1])
            print(w)
            # print('training ', i)
            # counter = 0

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self.sigmoid(output)
                w += self.learning_rate * errors.T.dot(X)
                
                # counter += 1
                # if counter // 10 == 0:
                #     print(sum(errors**2) / 2.0)
            self.w_.append((w, i))

        return self

    def getW(self):
        return self.w_

    def predictOne(self, x):
        return max((x.dot(w), c) for w, c in self.w_)[1]

    def predict(self, X):
        return np.array([self.predictOne(i) for i in np.insert(X, 0, 1, axis=1)])

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

[X, Y] = read_csv_file(sys.argv[1])

logreg = LogisticRegression(n_iter=1000)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

logreg.fit(X, Y)
# print(logreg.getW())
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])