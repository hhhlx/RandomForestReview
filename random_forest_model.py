from matplotlib import pyplot as plt
from sklearn import datasets
import math

import numpy as np


class Node():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        self._leaf_value_calculation = None
        self.one_dim = None
        self.loss = loss

    def fit(self, X, y, loss=None):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                unique_values = np.unique(np.expand_dims(X[:, feature_i], axis=1))

                for threshold in unique_values:
                    split_func = lambda sample: sample[feature_i] >= threshold if isinstance(threshold, int) or isinstance(threshold, float) else lambda sample: sample[feature_i] == threshold
                    Xy1 = np.array([sample for sample in Xy if split_func(sample)])
                    Xy2 = np.array([sample for sample in Xy if not split_func(sample)])

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        impurity = self._impurity_calculation(y, Xy1[:, n_features:], Xy2[:, n_features:])

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                            }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return Node(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        return Node(value=self._leaf_value_calculation(y))

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]

        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred


class Plot:
    def __init__(self):
        self.cmap = plt.get_cmap('viridis')

    def _transform(self, X, dim):
        eigenvalues, eigenvectors = np.linalg.eig(np.array((1 / (np.shape(X)[0] - 1)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0)), dtype=float))
        return X.dot(np.atleast_1d(eigenvectors[:, eigenvalues.argsort()[::-1]])[:, :dim])

    def plot2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self._transform(X, dim=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        class_distr = []

        y = np.array(y).astype(int)

        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        if not legend_labels is None:
            plt.legend(class_distr, legend_labels, loc=1)

        if title:
            if accuracy:
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % (100 * accuracy), fontsize=10)
            else:
                plt.title(title)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()


class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        return calculate_entropy(y) - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)

def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    entropy = 0
    for label in np.unique(y):
        p = len(y[y == label]) / len(y)
        entropy += -p * log2(p)
    return entropy

class RandomForest:
    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        for _ in range(self.n_estimators):
            self.trees.append(ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=self.min_gain, max_depth=self.max_depth))

    def fit(self, X, Y):
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            self.trees[i].fit(sub_X[:, idx], sub_Y)
            self.trees[i].feature_indices = idx
            print(f"tree {i} fit complete")

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            y_preds.append(self.trees[i].predict(X[:, self.trees[i].feature_indices]))
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred

    def get_bootstrap_data(self, X, Y):
        m = X.shape[0]
        Y = Y.reshape(m, 1)

        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            bootstrap_X_Y = X_Y[np.random.choice(m, m, replace=True), :]
            data_sets.append([bootstrap_X_Y[:, :-1], bootstrap_X_Y[:, -1:]])
        return data_sets


if __name__ == "__main__":
    data = datasets.load_digits()
    X = data.data
    y = data.target

    np.random.seed(2)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    split_i = len(y) - int(len(y) // (1 / 0.4))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    clf = RandomForest(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.sum(y_test == y_pred, axis=0) / len(y_test)

    print("Accuracy:", accuracy)

    Plot().plot2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)
