import numpy as np

import decisionTree

class AdaBoostedTree:
    def fit(self, features:np.ndarray, labels:np.ndarray, numTrees:int=10):
        num_rows, num_cols = features.shape
        self._trees = []
        w = np.ones(num_rows)
        for i in range(numTrees):
            self._trees.append(decisionTree.DecisionTree(decisionTree.SplitCondition.WEIGHTED_ZERO_ONE, maxDepth=1))