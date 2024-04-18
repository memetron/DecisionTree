from enum import Enum
import numpy as np
import statistics

from sklearn import datasets
from sklearn.model_selection import train_test_split

import splitFunctions


class SplitCondition(Enum):
    CONDITIONAL_ENTROPY = 1
    GINI_SPLIT = 2
    MSE = 3
    WEIGHTED_ZERO_ONE = 4


CLASSIFIERS = [SplitCondition.GINI_SPLIT, SplitCondition.CONDITIONAL_ENTROPY]


class DecisionTreeNode:
    _minData: int
    _maxDepth: int
    _depth: int
    _isLeaf: bool
    _splitCondition: SplitCondition
    _isClassifier: bool
    _splitProportion: float

    # returns (-1, -1) if no splits possible with min size
    def best_split(self, features: np.ndarray, labels: np.ndarray, indices,weights:np.ndarray=None):
        num_rows, num_cols = features.shape

        if self._splitProportion != 1.0:
            indices = np.random.choice(indices, size=int(len(indices) * self._splitProportion), replace=False)

        features_subset = features[indices]
        labels_subset = labels[indices]

        best_split_info = (float('inf'), -1, -1)
        for feature_index in range(num_cols):
            unique_feature_vals = np.unique(features_subset[:, feature_index])
            candidate_thresholds = (unique_feature_vals[:-1] + unique_feature_vals[1:]) / 2
            for threshold in candidate_thresholds:
                left_subset, right_subset = splitFunctions.split(features, indices, feature_index, threshold)
                if np.size(left_subset) >= self._minData and np.size(right_subset) >= self._minData:
                    # Various split conditions
                    if self._splitCondition == SplitCondition.CONDITIONAL_ENTROPY:
                        entropy = splitFunctions.conditionalEntropy(left_subset, right_subset, labels)
                        if entropy < best_split_info[0]:
                            best_split_info = (entropy, feature_index, threshold)

                    elif self._splitCondition == SplitCondition.GINI_SPLIT:
                        gini_impurity = splitFunctions.giniSplit(left_subset, right_subset, labels)
                        if gini_impurity < best_split_info[0]:
                            best_split_info = (gini_impurity, feature_index, threshold)

                    elif self._splitCondition == SplitCondition.MSE:
                        mse = splitFunctions.mse_split(left_subset, right_subset, labels)
                        if mse < best_split_info[0]:
                            best_split_info = (mse, feature_index, threshold)

                    elif self._splitCondition == SplitCondition.WEIGHTED_ZERO_ONE:
                        loss = splitFunctions.weighted_zero_one_split(left_subset, right_subset, labels, weights)
                        if loss < best_split_info[0]:
                            best_split_info = (loss, feature_index, threshold)

        return best_split_info[1], best_split_info[2]

    def __init__(self, features: np.ndarray, labels: np.ndarray, depth: int, splitCondition: SplitCondition,
                 indices, minData: int = 3, maxDepth: int = 5, splitProportion: float = 1.0):
        self._depth = depth
        self._splitCondition = splitCondition
        self._isClassifier = splitCondition in CLASSIFIERS
        self._minData = minData
        self._maxDepth = maxDepth
        self._splitProportion = splitProportion

        if self._depth >= self._maxDepth:
            self._isLeaf = True
            self.label = statistics.mode(labels[indices])
        # homogeneous case
        elif len(np.unique(labels[indices])) == 1:
            self._isLeaf = True
            self.label = labels[indices][0]
        else:
            self._isLeaf = False
            index, threshold = self.best_split(features, labels, indices)
            if (index, threshold) == (-1, -1):
                self._isLeaf = True
                if self._isClassifier:
                    self.label = statistics.mode(labels[indices])
                else:
                    self.label = statistics.mean(labels[indices])
            else:
                self._index = index
                self._threshold = threshold
                leftIndices, rightIndices = splitFunctions.split(features, indices, index, threshold)
                self._left = DecisionTreeNode(features, labels, self._depth + 1, self._splitCondition, leftIndices,
                                              maxDepth=self._maxDepth, minData=self._minData,
                                              splitProportion=self._splitProportion)
                self._right = DecisionTreeNode(features, labels, self._depth + 1, self._splitCondition, rightIndices,
                                              maxDepth=self._maxDepth, minData=self._minData,
                                               splitProportion=self._splitProportion)
                self._isLeaf = False

    def predict(self, featureVector: np.ndarray):
        if self._isLeaf:
            return self.label
        else:
            if featureVector[self._index] < self._threshold:
                return self._left.predict(featureVector)
            else:
                return self._right.predict(featureVector)

    def toString(self, featureNames, level=0):
        indent = "|   " * level
        if self._isLeaf:
            return f"{indent}|--- class: {self.label}"
        else:
            left_str = self._left.toString(featureNames, level + 1)
            right_str = self._right.toString(featureNames, level + 1)
            return f"{indent}{featureNames[self._index]} < {self._threshold}\n{left_str}\n{right_str}"


class DecisionTree:
    _root: DecisionTreeNode
    _splitCondition: SplitCondition
    _splitProportion: float
    _minData: int
    _maxDepth: int

    def __init__(self, splitCondition: SplitCondition,
                 minData: int = 3, maxDepth: int = 5, splitProportion: float = 1.0):
        self._minData = minData
        self._maxDepth = maxDepth
        self._splitCondition = splitCondition
        self._splitProportion = splitProportion

    def fit(self, features: np.ndarray, labels: np.ndarray):
        num_rows, num_cols = features.shape
        self._root = DecisionTreeNode(features, labels, 0, self._splitCondition, np.arange(0, num_rows),
                                      minData=self._minData, maxDepth=self._maxDepth, splitProportion=self._splitProportion)

    def predict(self, features):
        return self._root.predict(features)

    def predict_mult(self, features):
        prediction_array = []
        for f in features:
            prediction_array.append(self.predict(f))
        return np.array(prediction_array)

    def toString(self, featureNames):
        return self._root.toString(featureNames)

def wine_data():
    wine = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
    return X_train, y_train, X_test, y_test

def test1():
    X_train, y_train, X_test, y_test = wine_data()
    model = DecisionTree(SplitCondition.CONDITIONAL_ENTROPY)
    model.fit(X_train, y_train)
    correctCount = 0
    incorrectCount = 0
    for i in range(len(y_test)):
        if model.predict(X_test[i]) == y_test[i]:
            correctCount += 1
        else:
            incorrectCount += 1
    print(f"Model accuracy = {correctCount / (correctCount + incorrectCount)}")

def main():
    test1()

if __name__ == "__main__":
    test1()