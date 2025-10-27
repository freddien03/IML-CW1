from matplotlib.pylab import default_rng
import numpy as np
from decision_tree import Decision_Tree_Classifier


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


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.
    
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds

def cross_validation(n_folds, data, random_generator=default_rng()):
    n_instances = data.shape[0]
    accuracies = np.zeros(n_folds, dtype=float)
    dt_classifier = Decision_Tree_Classifier()

    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, n_instances, random_generator)):
        train_data = data[train_indices, :]
        test_data = data[test_indices, :]

        # train decision tree
        dt_classifier.fit(train_data)
        # do shit with it