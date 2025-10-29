import numpy as np
from node import Node, Leaf

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.depth = 0

    def fit(self, data):
        # fits the tree from full (X|y) data and records depth
        self.tree, self.depth = self.decision_tree_learning(data, 0)

    def decision_tree_learning(self, data, depth):
        # recursively grows the tree and stops if all labels are identical
        if len(np.unique(data[:, -1])) == 1:
            return (Leaf(data[:, -1][0]), depth)

        attr, val, l_data, r_data = self.find_split(data)
        l_branch, l_depth = self.decision_tree_learning(l_data, depth + 1)
        r_branch, r_depth = self.decision_tree_learning(r_data, depth + 1)
        return (Node(attr, val, l_branch, r_branch), max(l_depth, r_depth))

    def find_split(self, data):
        # searches all attributes/thresholds, picks split with minimum remainder
        best_attr = None
        best_val = None
        best_left = None
        best_right = None
        best_rem = np.inf

        for j in range(data.shape[1] - 1):
            sorted_data = data[data[:, j].argsort()]

            i = 1
            while i < len(sorted_data):
                prev = sorted_data[i - 1, j]
                while i < len(sorted_data) and sorted_data[i, j] == prev:
                    i += 1
                if i == len(sorted_data):
                    break

                left = sorted_data[:i]
                right = sorted_data[i:]
                rem = self.calc_rem(left, right)

                if rem < best_rem:
                    best_attr = j
                    best_val = (sorted_data[i - 1, j] + sorted_data[i, j]) / 2
                    best_left = left
                    best_right = right
                    best_rem = rem

                i += 1

        return (best_attr, best_val, best_left, best_right)

    def calc_H(self, data):
        # entropy of labels in data
        labels = data[:, -1]
        n = len(labels)
        total = 0.0
        _, counts = np.unique(labels, return_counts=True)
        for cnt in counts:
            p = cnt / n
            total += p * np.log2(p)
        return -total

    def calc_rem(self, left, right):
        # calculates weighted remainder of left/right partitions
        ln = len(left)
        rn = len(right)
        return (ln / (ln + rn)) * self.calc_H(left) + (rn / (ln + rn)) * self.calc_H(right)
