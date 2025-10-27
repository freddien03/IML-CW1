import numpy as np
from node import Node, Leaf

clean = np.loadtxt("wifi_db/clean_dataset.txt")
noisy = np.loadtxt("wifi_db/noisy_dataset.txt")

class Decision_Tree:

    def __init__(self):
        pass
        
    def decision_tree_learning(self, data: np.array, depth: int):
        if np.all(data[:,-1] == data[:,-1][0]):
            return (Leaf(data[:,-1][0]), depth)
        
        attr, val, l_data, r_data = self.find_split()
        l_branch, l_depth = self.decision_tree_learning(l_data, depth+1)
        r_branch, r_depth = self.decision_tree_learning(r_data, depth+1)
        return (Node(attr, val, l_branch, r_branch), max(l_depth, r_depth))

    

    def find_split(self, data):
        return