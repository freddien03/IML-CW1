import numpy as np
from node import Node, Leaf

clean = np.loadtxt("wifi_db/clean_dataset.txt")
noisy = np.loadtxt("wifi_db/noisy_dataset.txt")

class Decision_Tree_Classifier:

    def __init__(self):
        self.tree = None

    def fit(self, data):
        self.tree, self.depth = self.decision_tree_learning(data, 0)

    def predict(self, data):
        return np.array([self.predict_one(x) for x in data])

    def predict_one(self, x):
        tree = self.tree
        while isinstance(tree, Node):
            tree = tree.right if x[tree.attr] > tree.val else tree.left
        return tree.label

    def decision_tree_learning(self, data, depth):
        if len(np.unique(data[:,-1])) == 1:
            return (Leaf(data[:,-1][0]), depth)
        
        attr, val, l_data, r_data = self.find_split(data)
        l_branch, l_depth = self.decision_tree_learning(l_data, depth+1)
        r_branch, r_depth = self.decision_tree_learning(r_data, depth+1)

        return (Node(attr, val, l_branch, r_branch), max(l_depth, r_depth))

    

    def find_split(self, data):
        attr = None
        val = None
        l_data = None
        r_data = None
        min_rem = np.inf
        for j in range(data.shape[1] - 1):
            sorted_data = data[data[:,j].argsort()]

            i = 1
            while i < len(sorted_data):
                old = sorted_data[i-1, j]
                while i < len(sorted_data) and sorted_data[i, j] == old:
                    i += 1
                if i == len(sorted_data):
                    break

                left = sorted_data[:i]
                right = sorted_data[i:]
                rem = self.calc_rem(left, right)

                if rem < min_rem:
                    attr = j
                    val = (sorted_data[i - 1, j] + sorted_data[i, j]) / 2
                    l_data = left
                    r_data = right
                    min_rem = rem

                i += 1

        return (attr, val, l_data, r_data)
        

    
    def calc_H(self, data):
        labels = data[:,-1]
        N = len(labels)
        total = 0
        uniques, counts = np.unique(labels, return_counts=True)
        for l, n in zip(uniques, counts):
            total += (n / N) * np.log2(n / N)
        return -total
    
    def calc_rem(self, left, right):
        ln = len(left)
        rn = len(right)

        return (ln / (ln + rn)) * self.calc_H(left) + (rn / (ln + rn)) * self.calc_H(right)
