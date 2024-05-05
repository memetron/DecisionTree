import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import decisionTree
import splitFunctions


class AdaBoostedTree:
    _alpha: np.ndarray

    def __init__(self, numtrees: int = 10):
        self._numtrees = numtrees
        self._alpha = np.ones(numtrees)

    def fit(self, features: np.ndarray, labels: np.ndarray):
        num_rows, num_cols = features.shape
        self._trees = []
        w = np.ones(num_rows) / num_rows

        for i in range(self._numtrees):
            self._trees.append(
                decisionTree.DecisionTree(splitter=splitFunctions.WeightedZeroOneSplitter(np.ndarray.copy(w)), maxDepth=1, minData=1))
            self._trees[i].fit(features, labels)
            predictions = self._trees[i].predict_mult(features)

            # magic adaboost formulas to get new weights
            err = np.sum((predictions != labels) * w)
            self._alpha[i] = 1 / 2 * math.log((1 - err) / err, math.e)
            w = w * (math.e ** (- labels * predictions * self._alpha[i])) / (2 * math.sqrt(err * (1 - err)))

    def predict(self, features):
        predictions = np.array([tree.predict(features) for tree in self._trees])
        return 1 if np.sum(predictions * self._alpha) > 0 else -1

    def predict_mult(self, features):
        prediction_array = []
        for f in features:
            prediction_array.append(self.predict(f))
        return np.array(prediction_array)

def test():
    X, y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=1)
    y = np.array([-1 if i % 2 == 0 else 1 for i in y])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    blob1 = X_train[y_train == -1]
    blob2 = X_train[y_train == 1]
    plt.scatter(blob1[:, 0], blob1[:, 1], marker='+', color='r', label='class0')
    plt.scatter(blob2[:, 0], blob2[:, 1], marker='_', color='b', label='class1')

    # Train Random Forest model
    model = AdaBoostedTree(numtrees=50)
    model.fit(X_train, y_train)
    model.predict_mult(X_train)

    # plt.scatter(np.array(disagreeances)[:, 0], np.array(disagreeances)[:, 1], marker='p', color='y',
    #             label='disagreement in forest')
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                         np.arange(y_min, y_max, 0.005))
    Z = model.predict_mult(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.legend()
    plt.show()

def main():
    test()

if __name__ == "__main__":
    main()