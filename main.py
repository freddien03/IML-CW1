import numpy as np

from decision_tree import Decision_Tree_Classifier
from evaluation import evaluate, cross_validation

def run_dataset(name, data, labels=None, n_folds=10, seed=0):
    if labels is None:
        labels = np.array([1., 2., 3., 4.])

    print(f"===== {name} ({n_folds}-fold) =====")
    metrics = cross_validation(n_folds, data, labels=labels, seed=seed)

    confusion = metrics["confusion_matrix"]
    print("Confusion matrix (rows=true, cols=pred):\n")
    print(confusion)
    print()
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print("Precision :", np.round(metrics["precision"], 4))
    print("Recall    :", np.round(metrics["recall"], 4))
    print("F1        :", np.round(metrics["f1"], 4))
    print(f"Macro-P   : {metrics['macro_precision']:.4f}")
    print(f"Macro-R   : {metrics['macro_recall']:.4f}")
    print(f"Macro-F1  : {metrics['macro_f1']:.4f}")
    print("Fold accuracies:", np.round(metrics["fold_accuracies"], 4),
          "| mean =", round(metrics["fold_accuracies"].mean(), 4))

    np.savetxt(f"{name.lower()}_confusion_matrix.txt", confusion, fmt="%d")

if __name__ == "__main__":
    clean = np.loadtxt("wifi_db/clean_dataset.txt")
    noisy = np.loadtxt("wifi_db/noisy_dataset.txt")

    clf = Decision_Tree_Classifier()
    clf.fit(clean)
    print("Spec evaluate(clean, trained_tree):",
          round(evaluate(clean, clf.tree), 4), "\n")

    run_dataset("CLEAN", clean, labels=np.array([1., 2., 3., 4.]), n_folds=10, seed=0)
    print()
    run_dataset("NOISY", noisy, labels=np.array([1., 2., 3., 4.]), n_folds=10, seed=0)
