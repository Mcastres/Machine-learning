import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import sys
import math
import multiprocessing
import time

# Read the csv dataset and return an array of it
def read_csv_file(file):
    dataset = pd.read_csv(sys.argv[1])
    x = dataset.iloc[:, 5:].values
    y = dataset.iloc[:, 1].values
    y = np.unique(y, return_inverse=True)[1].tolist()
    x[:, 0] = np.unique(x[:, [0]], return_inverse=True)[1].tolist()
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(x)
    x = imputer.transform(x)
    return [x, y]

def loss( h , y, m):
    return sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / -m

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gd(index ,y_copy, X, learning_rate, dic):
    m = y_copy.shape[0]
    w = np.ones(X.shape[1])

    iter = 0
    cost = 0
    for _ in range(10000):
        iter += 1
        h = sigmoid(X.dot(w))
        w -= learning_rate * ((h - y_copy).dot(X) / m)
        if (iter % 1000 == 0):
            cost = loss(h, y_copy, m)
            print(str(iter) + " cost : " + str(cost) + " index : " + str(index))
    dic[index] = w

def mini_batch_sgd(index ,y_copy, X, learning_rate, dic):
    m = y_copy.shape[0]
    w = np.ones(X.shape[1])
    print(w)
    exit(0)
    batch_size = 64
    cost = 0
    iter = 0
    for _ in range(1000):
        indices = np.random.permutation(m)
        X = X[indices]
        Y = y_copy[indices]
        iter += 1
        i = np.random.randint(0, y_copy.shape[0] - batch_size)
        print(i)
        X_i = X[i:i+batch_size]
        Y_i = Y[i:i+batch_size]
        h = sigmoid(X_i.dot(w))
        cost = loss(h, Y_i, m)
        w -= learning_rate * ((h - Y_i).dot(X_i) / m)
        if (iter % 1000 == 0):
            print(str(iter) + " cost : " + str(cost) + " index : " + str(index))
    dic[index] = w

class LogisticRegression(object):

    def create_csv_file(data):
        data.to_csv("weights", encoding='utf-8')

    def __init__(self, learning_rate=0.001, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 0, axis=1)
        self.weight = []
        manager = multiprocessing.Manager()
        dic = manager.dict()
        jobs = []
        for i in np.unique(y):
            y_copy = np.array([1 if c == i else 0 for c in y])
            p = multiprocessing.Process(target=mini_batch_sgd, args=(i, y_copy, np.copy(X), self.learning_rate, dic))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        for key in dic:
            self.weight.append((dic[key], key))
        return self

    def getW(self):
        return self.weight

    def predictOne(self, x):    
        return max((x.dot(w), c) for w, c in self.weight)[1]

    def predict(self, X):
        return np.array([self.predictOne(i) for i in np.insert(X, 0, 0, axis=1)])

[X, Y] = read_csv_file(sys.argv[1])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

logreg = LogisticRegression(n_iter=1000)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
print(logreg.getW())
print(accuracy_score(y_test, y_pred))