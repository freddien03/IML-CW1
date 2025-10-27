import numpy as np

from node import Node

def predict_one(tree, x):
    while isinstance(tree, Node):
        tree = tree.right if x[tree.attr] > tree.val else tree.left
    return tree.label

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

def evaluate(test_db, trained_tree):
    X, y = test_db[:, :-1], test_db[:, -1]
    y_hat = predict(trained_tree, X)
    
    return float(np.mean(y_hat == y))

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    L = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((L, L), dtype=float)
    
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    
    return cm, np.array(labels)