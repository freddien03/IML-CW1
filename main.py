import numpy as np
from decision_tree import DecisionTreeClassifier
from evaluation import evaluate, cross_validation
from visualiser import save_tree_png

RNG_SEED = 67  # fixed seed for deterministic k-fold splits, change if needed

def run_dataset(name, data, labels=None, n_folds=10, seed=37):
    # runs k-fold CV, prints metrics, and saves the confusion matrix
    if labels is None:
        labels = np.array([1., 2., 3., 4.])

    print(f"=~*~= {name} ({n_folds}-fold) =~*~=")
    metrics = cross_validation(n_folds, data, labels=labels, seed=seed)

    confusion = metrics["confusion_matrix"]
    print("Confusion matrix (rows=true, cols=pred):\n")
    print(confusion)
    print()
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print("Precision :", np.round(metrics["precision"], 4))
    print("Recall    :", np.round(metrics["recall"], 4))
    print("F1        :", np.round(metrics["f1"], 4))
    print(f"Macro-P   : {metrics['macro_precision']:.6f}")
    print(f"Macro-R   : {metrics['macro_recall']:.6f}")
    print(f"Macro-F1  : {metrics['macro_f1']:.6f}")
    print(
        "Fold accuracies:", np.round(metrics["fold_accuracies"], 4),
        "| mean =", round(metrics["fold_accuracies"].mean(), 4),
    )

    np.savetxt(f"{name.lower()}_confusion_matrix.txt", confusion, fmt="%d")

if __name__ == "__main__":
    # load datasets and train once on full datasets
    clean = np.loadtxt("wifi_db/clean_dataset.txt")
    noisy = np.loadtxt("wifi_db/noisy_dataset.txt")

    clf = DecisionTreeClassifier()
    clf.fit(clean)

    # save a visualisation of the tree trained on the full clean dataset
    # change these parameters to get a different layou
    save_tree_png(
        clf.tree,
        path="clean_tree.png",
        figsize=(14, 7),
        hsep=1.7,
        vstep=1.7,
        node_fs=9,
        leaf_fs=9,
        box_pad=0.18,
        decimals=1,
        dpi=140,
        show=False
    )
    print("Saved visualisation to clean_tree.png\n")

    # run a 10-fold CV on clean and noisy datasets
    run_dataset("CLEAN", clean, labels=np.array([1., 2., 3., 4.]), n_folds=10, seed=RNG_SEED)
    print()
    run_dataset("NOISY", noisy, labels=np.array([1., 2., 3., 4.]), n_folds=10, seed=RNG_SEED)
