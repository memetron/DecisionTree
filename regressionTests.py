import numpy as np
import matplotlib.pyplot as plt
import randomForest
import decisionTree

def create_polynomial_data(n_samples=100, degree=3, noise=0.1):
    # Generate random data
    np.random.seed(0)
    x = np.sort(5 * np.random.rand(n_samples))
    y_true = np.sin(x) + np.random.normal(0, noise, n_samples)

    # Create polynomial features
    X = np.vstack([x ** i for i in range(1, degree + 1)]).T

    return X, y_true


def plot_predictions_vs_actual(x, y_true, y_pred):
    plt.scatter(x, y_true, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Predicted')
    plt.xlabel('feature')
    plt.ylabel('label')
    plt.title("random forest regression")
    plt.legend()
    plt.show()


def polynomial_regression_with_decision_tree():
    # Create polynomial data
    X, y_true = create_polynomial_data(degree=3)

    # Fit Decision Tree Regressor
    model = randomForest.RandomForest(decisionTree.SplitCondition.MSE, numTrees=100, splitProportion=0.5)
    model.fit(X, y_true)

    # Predictions
    y_pred = model.predict_mult(X)

    # Plot
    plot_predictions_vs_actual(X[:, 0], y_true, y_pred)


def main():
    polynomial_regression_with_decision_tree()


if __name__=="__main__":
    main()