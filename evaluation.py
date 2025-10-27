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

def confusion_matrix(y_true, y_pred, class_labels=None):
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_true, y_pred)))

    C = len(class_labels)
    confusion = np.zeros((C, C), dtype=int)

    for i, label in enumerate(class_labels):
        indices = (y_true == label)
        preds = y_pred[indices]
        unique_labels, counts = np.unique(preds, return_counts=True)
        freq = dict(zip(unique_labels, counts))
        for j, cl in enumerate(class_labels):
            confusion[i, j] = freq.get(cl, 0)

    return confusion

def precision(y_true, y_pred, class_labels=None):
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_true, y_pred)))
    
    confusion = confusion_matrix(y_true, y_pred, class_labels=class_labels)

    C = confusion.shape[0]
    p = np.zeros(C, dtype=float)
    
    for i in range(C):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        denom = tp + fp
        p[i] = tp / denom if denom else 0.0

    macro_p = float(np.mean(p)) if C else 0.0
    return p, macro_p

def recall(y_true, y_pred, class_labels=None):
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_true, y_pred)))
    
    confusion = confusion_matrix(y_true, y_pred, class_labels=class_labels)

    C = confusion.shape[0]
    r = np.zeros(C, dtype=float)
    
    for i in range(C):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        denom = tp + fn
        r[i] = tp / denom if denom else 0.0

    macro_r = float(np.mean(r)) if C else 0.0
    return r, macro_r

def f1_score(y_true, y_pred, class_labels=None):
    p, _ = precision(y_true, y_pred, class_labels)
    r, _ = recall(y_true, y_pred, class_labels)

    C = len(p)
    f1 = np.zeros(C, dtype=float)
    for i in range(C):
        denom = p[i] + r[i]
        f1[i] = 2 * p[i] * r[i] / denom if denom else 0.0

    macro_f1 = float(np.mean(f1)) if C else 0.0
    return f1, macro_f1