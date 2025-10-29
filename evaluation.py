import numpy as np
from numpy.random import default_rng
from decision_tree import DecisionTreeClassifier
from node import Node

def _predict_one(tree, x):
    # traverses a trained tree to predict one sample 
    # note: this is a helper function for predict
    while isinstance(tree, Node):
        tree = tree.right if x[tree.attr] > tree.val else tree.left
    return tree.label

def predict(tree, X):
    # vectorised tree prediction over a matrix X
    return np.array([_predict_one(tree, x) for x in X])

def evaluate(test_db, trained_tree):
    # evaluate accuracy of a trained tree on test data
    X, y = test_db[:, :-1], test_db[:, -1]
    y_hat = predict(trained_tree, X)
    return float(np.mean(y_hat == y))

def confusion_matrix(y_true, y_pred, labels=None):
    # builds confusion matrix with a fixed label order
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    C = len(labels)
    cm = np.zeros((C, C), dtype=int)

    for i, lab in enumerate(labels):
        idx = (y_true == lab)
        preds = y_pred[idx]
        uniq, cnt = np.unique(preds, return_counts=True)
        freq = dict(zip(uniq, cnt))
        for j, lab_j in enumerate(labels):
            cm[i, j] = freq.get(lab_j, 0)

    return cm

def precision(y_true, y_pred, labels=None):
    # calculates per-class precision and macro-average
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    C = cm.shape[0]
    prec = np.zeros(C, dtype=float)
    for i in range(C):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        denom = tp + fp
        prec[i] = tp / denom if denom else 0.0

    macro_prec = float(np.mean(prec)) if C else 0.0
    return prec, macro_prec

def recall(y_true, y_pred, labels=None):
    # calculates per-class recall and macro-average
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    C = cm.shape[0]
    rec = np.zeros(C, dtype=float)
    for i in range(C):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        denom = tp + fn
        rec[i] = tp / denom if denom else 0.0

    macro_rec = float(np.mean(rec)) if C else 0.0
    return rec, macro_rec

def f1_score(y_true, y_pred, labels=None):
    # calculates per-class F1 and macro-average
    prec, _ = precision(y_true, y_pred, labels)
    rec, _ = recall(y_true, y_pred, labels)

    C = len(prec)
    f1 = np.zeros(C, dtype=float)
    for i in range(C):
        denom = prec[i] + rec[i]
        f1[i] = 2 * prec[i] * rec[i] / denom if denom else 0.0

    macro_f1 = float(np.mean(f1)) if C else 0.0
    return f1, macro_f1

def k_fold_split(n_splits, n_instances, rng=default_rng()):
    # randomly splits indices [0..n_instances) into n_splits folds
    shuffled = rng.permutation(n_instances)
    return np.array_split(shuffled, n_splits)

def train_test_k_fold(n_folds, n_instances, rng=default_rng()):
    # yields train/test index arrays for each fold
    splits = k_fold_split(n_folds, n_instances, rng)
    folds = []
    for k in range(n_folds):
        test_idx = splits[k]
        train_idx = np.hstack(splits[:k] + splits[k+1:])
        folds.append([train_idx, test_idx])
    return folds

def cross_validation(n_folds, data, labels=None, seed=37):
    # k-fold CV: train on each fold, collect metrics and confusion matrix
    rng = default_rng(seed)
    folds = train_test_k_fold(n_folds, data.shape[0], rng)
    if labels is None:
        labels = np.array([1., 2., 3., 4.])

    y_true_all, y_pred_all = [], []
    fold_acc = np.zeros(n_folds, dtype=float)

    for i, (train_idx, test_idx) in enumerate(folds):
        train, test = data[train_idx], data[test_idx]
        clf = DecisionTreeClassifier()
        clf.fit(train)
        X_test, y_test = test[:, :-1], test[:, -1]
        y_hat = predict(clf.tree, X_test)

        fold_acc[i] = np.mean(y_hat == y_test)
        y_true_all.append(y_test)
        y_pred_all.append(y_hat)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    acc = float(np.trace(cm) / cm.sum())
    prec, macro_prec = precision(y_true_all, y_pred_all, labels=labels)
    rec, macro_rec = recall(y_true_all, y_pred_all, labels=labels)
    f1, macro_f1 = f1_score(y_true_all, y_pred_all, labels=labels)

    return {
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "fold_accuracies": fold_acc,
    }
