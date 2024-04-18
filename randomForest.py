import numpy as np
import statistics

import decisionTree
from decisionTree import DecisionTree, SplitCondition


class RandomForest:
    _splitCondition: SplitCondition
    _numTrees: int
    _minData: int
    _maxDepth: int
    _splitProportion: float

    def __init__(self, splitCondition: SplitCondition,
                 numTrees: int = 10, minData: int = 3, maxDepth: int = 100, splitProportion: float = 0.7):
        self._splitCondition = splitCondition
        self._numTrees = numTrees
        self._minData = minData
        self._maxDepth = maxDepth
        self._splitProportion = splitProportion
        self._isClassifier = splitCondition in decisionTree.CLASSIFIERS

        self.trees = []
        for i in range(numTrees):
            newTree = DecisionTree(splitCondition, minData=minData, maxDepth=maxDepth, splitProportion=splitProportion)
            self.trees.append(newTree)

    def fit(self, features: np.ndarray, labels: np.ndarray):
        num_rows, num_cols = np.shape(features)
        for tree in self.trees:
            indices = np.random.choice(num_rows, size=num_rows, replace=True)
            tree.fit(features[indices], labels[indices])

    def predict(self, features):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(features))
        return statistics.mode(predictions) if self._isClassifier else statistics.mean(predictions)

    def predict_mult(self, features):
        prediction_array = []
        for f in features:
            prediction_array.append(self.predict(f))
        return np.array(prediction_array)
