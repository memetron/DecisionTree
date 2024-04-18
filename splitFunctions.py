import statistics

import numpy as np


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


class WeightedZeroOneSplitter(Splitter):
    _weights: np.ndarray

    def __init__(self, weights: np.ndarray):
        self._weights = weights

    def weighted_zero_one(self, labels: np.ndarray):
        target = statistics.mode(labels)
        misses = labels != target
        return np.sum(misses * self._weights)

    def splitLoss(self, left, right, labels) -> float:
        leftLabels = labels[left]
        rightLabels = labels[right]
        return self.weighted_zero_one(leftLabels) + self.weighted_zero_one(rightLabels)

    def isClassifier(self) -> bool:
        return True
