from matplotlib import pyplot as plt

import decisionTree
import numpy as np

class GradientBoostedTree:
    _numTrees: int
    _learningRate: float
    _minData: int

    def __init__(self, numTrees: int, learningRate: float = 0.1, minData=1):
        self._minData = minData
        self._numTrees = numTrees
        self._learningRate = learningRate
        self._trees = []

    def fit(self, features: np.ndarray, labels: np.ndarray):
        for i in range(self._numTrees):
            # curr prediction of model is the sum of its trees
            currPrediction = np.sum(np.array([tree.predict_mult(features) for tree in self._trees]), axis=0)
            residual = labels - currPrediction
            # fit against the residual using a weak model
            newTree = decisionTree.DecisionTree(minData=self._minData, maxDepth=2, splitCondition=decisionTree.SplitCondition.MSE)
            newTree.fit(features, residual)
            self._trees.append(newTree)
            # self._graph_residual(features, residual, i)

    def predict(self, features: np.ndarray):
        # curr prediction of model is the sum of its trees
        return np.sum(np.array([tree.predict(features) for tree in self._trees]))

    def predict_mult(self, features):
        prediction_array = []
        for f in features:
            prediction_array.append(self.predict(f))
        return np.array(prediction_array)

    def _graph_residual(self, features, residual, n):
        y_pred = self._trees[n].predict_mult(features)
        self._plot_predictions_vs_actual(features[:, 0], residual, y_pred, n)

    def _plot_predictions_vs_actual(self, x, y_true, y_pred, n):
        plt.scatter(x, y_true, color='blue', label='Actual')
        plt.plot(x, y_pred, color='red', label='Predicted')
        plt.xlabel('feature')
        plt.ylabel('label')
        plt.title(f"residual n = {n}")
        plt.legend()
        plt.show()
