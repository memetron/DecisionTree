import statistics

import numpy as np


def split(features: np.ndarray, indices, index: int, threshold: float):
    left = features[indices][:, index] < threshold
    right = ~left
    left_indices = indices[left]
    right_indices = indices[right]
    return left_indices, right_indices


def entropy(labels: np.ndarray):
    elements, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -1 * np.sum(probabilities * np.log(probabilities))


def conditionalEntropy(left, right, labels: np.ndarray):
    numLeft = np.size(left)
    numRight = np.size(right)
    leftLabels = labels[left]
    rightLabels = labels[right]
    return (numLeft * entropy(leftLabels) + numRight * entropy(rightLabels)) / (numLeft + numRight)


def gini(labels):
    elements, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)


def giniSplit(left, right, labels):
    numLeft = left.sum()
    numRight = right.sum()
    total = numLeft + numRight
    leftLabels = labels[left]
    rightLabels = labels[right]
    return (numLeft * gini(leftLabels) + numRight * gini(rightLabels)) / total


def mse(labels: np.ndarray):
    return np.mean((labels - labels.mean()) ** 2)


def mse_split(left, right, labels):
    numLeft = left.sum()
    numRight = right.sum()
    leftLabels = labels[left]
    rightLabels = labels[right]
    return (numLeft * mse(leftLabels) + numRight * mse(rightLabels)) / (numLeft + numRight)


def weighted_zero_one(labels: np.ndarray, weights: np.ndarray):
    target = statistics.mode(labels)
    misses = labels != target
    return np.sum(misses * weights)


def weighted_zero_one_split(left, right, labels, weights):
    leftLabels = labels[left]
    rightLabels = labels[right]
    return weighted_zero_one(leftLabels, weights) + weighted_zero_one(rightLabels, weights)


