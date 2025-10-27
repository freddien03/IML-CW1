import numpy as np

from node import Node

def evaluate(test_db, trained_tree):
    X, y = test_db[:, :-1], test_db[:, -1]

    def _predict_one(x, t):
        while isinstance(t, Node):
            t = t.right if x[t.attr] > t.val else t.left
        return t.label

    y_hat = np.array([_predict_one(x, trained_tree) for x in X])
    return float(np.mean(y_hat == y))
