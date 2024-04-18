import statistics

from sklearn import datasets
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import decisionTree
import randomForest

import numpy as np

def main():
    pass

def depth_size_test(k, startingDepth, endingDepth, startingData, endingData):
    X_train, y_train, X_test, y_test = wine_data()
    accuracy_vs_depth_size(k, startingDepth, endingDepth, startingData, endingData, X_train, y_train, X_test, y_test)

def disagreement_test():
    X, y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=1.5)
    y = np.array([0 if i % 2 == 0 else 1 for i in y])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    display_disagreements(X_train, y_train)

def num_trees_test(k: int):
    X_train, y_train, X_test, y_test = wine_data()
    accuracy_vs_k(k, X_train, y_train, X_test, y_test)

def wine_data():
    wine = datasets.load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
    return X_train, y_train, X_test, y_test

# graph accuracy vs number of trees used in forest
def accuracy_vs_k(k: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    scores = []
    for i in range(1, k + 1):
        model = randomForest.RandomForest(i, 5, 5)
        model.fit(X_train, y_train)
        scores.append(accuracy(model, X_test, y_test))
    plt.plot(range(1, k + 1), scores, marker='o')
    plt.grid(True)
    plt.show()

def accuracy_vs_depth_size(k: int, startingDepth: int, endingDepth: int, startingData: int, endingData: int
                           , X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,y_test: np.ndarray):
    NUM_ATTEMPTS = 100
    scores = []

    for depth in range(startingDepth, endingDepth + 1):
        scores.append([])
        for data in range(startingData, endingData + 1):
            score = 0
            for i in range(NUM_ATTEMPTS):
                model = randomForest.RandomForest(k, data, depth)
                model.fit(X_train, y_train)
                score += accuracy(model, X_test, y_test)
            scores[depth - startingDepth].append(score / NUM_ATTEMPTS)

    plt.imshow(scores, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Accuracy')
    plt.xticks(range(endingData - startingData + 1), range(startingData, endingData + 1))
    plt.yticks(range(endingDepth - startingDepth + 1), range(startingDepth, endingDepth + 1))
    plt.xlabel('Size')
    plt.ylabel('Depth')
    plt.title('Accuracy vs Depth and Size')
    plt.show()

# displays disagreement among trees in model
def display_disagreements(X_train, y_train):
    blob1 = X_train[y_train == 0]
    blob2 = X_train[y_train == 1]
    plt.scatter(blob1[:, 0], blob1[:, 1], marker='+', color='r', label='class0')
    plt.scatter(blob2[:, 0], blob2[:, 1], marker='_', color='b', label='class1')

    # Train Random Forest model
    model = randomForest.RandomForest(decisionTree.SplitCondition.CONDITIONAL_ENTROPY, 1)
    model.fit(X_train, y_train)
    model.predict_mult(X_train)

    disagreeances = []
    for X in X_train:
        predictions = np.array([tree.predict(X) for tree in model.trees])
        mode = statistics.mode(predictions)
        v, c = np.unique(predictions, return_counts=True)
        agreeance = dict(zip(v,c))[mode] # number of trees that agree with mode
        if agreeance < len(model.trees):
            disagreeances.append(X.tolist())

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

def accuracy(model: randomForest.RandomForest, features: np.ndarray, labels: np.ndarray):
    predictions = model.predict_mult(features)
    return accuracy_score(labels, predictions)

if __name__ == "__main__":
    main()