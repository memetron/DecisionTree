import statistics

import numpy as np


# Abstract parent splitter
class Splitter:
    def split(self, features: np.ndarray, indices, index: int, threshold: float):
        left = features[indices][:, index] < threshold
        right = ~left
        left_indices = indices[left]
        right_indices = indices[right]
        return left_indices, right_indices

    def splitLoss(self, left, right, labels) -> float:
        raise NotImplemented()

    def isClassifier(self) -> bool:
        raise NotImplemented()


# Split using conditional entropy
class EntropySplitter(Splitter):
    def _entropy(self, labels: np.ndarray):
        elements, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return -1 * np.sum(probabilities * np.log(probabilities))

    def splitLoss(self, left, right, labels: np.ndarray) -> float:
        numLeft = np.size(left)
        numRight = np.size(right)
        leftLabels = labels[left]
        rightLabels = labels[right]
        return (numLeft * self._entropy(leftLabels) + numRight * self._entropy(rightLabels)) / (numLeft + numRight)

    def isClassifier(self) -> bool:
        return True


# Split using gini impurity
class GiniSplitter(Splitter):
    def _gini(self, labels):
        elements, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def splitLoss(self, left, right, labels) -> float:
        numLeft = left.sum()
        numRight = right.sum()
        total = numLeft + numRight
        leftLabels = labels[left]
        rightLabels = labels[right]
        return (numLeft * self._gini(leftLabels) + numRight * self._gini(rightLabels)) / total

    def isClassifier(self) -> bool:
        return True


# split using mean-squared error
class MSESplitter(Splitter):
    def _mse(self, labels: np.ndarray):
        return np.mean((labels - labels.mean()) ** 2)

    def splitLoss(self, left, right, labels) -> float:
        numLeft = left.sum()
        numRight = right.sum()
        leftLabels = labels[left]
        rightLabels = labels[right]
        return (numLeft * self._mse(leftLabels) + numRight * self._mse(rightLabels)) / (numLeft + numRight)

    def isClassifier(self) -> bool:
        return False


# Split using a weighted zero-one loss
class WeightedZeroOneSplitter(Splitter):
    _weights: np.ndarray

    def __init__(self, weights: np.ndarray):
        self._weights = weights

    def weighted_zero_one(self, labels: np.ndarray, weights):
        target = statistics.mode(labels)
        misses = labels != target
        return np.sum(misses * weights)

    def splitLoss(self, left, right, labels) -> float:
        leftLabels = labels[left]
        rightLabels = labels[right]
        leftWeights = self._weights[left]
        rightWeights = self._weights[right]
        return self.weighted_zero_one(leftLabels, leftWeights) + self.weighted_zero_one(rightLabels, rightWeights)

    def isClassifier(self) -> bool:
        return True
