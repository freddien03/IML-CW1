class Node:
    def __init__(self, attr, val, left, right):
        self.attr = attr
        self.val = val
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"({self.left}) <- {self.attr} < {self.val} -> ({self.right})"

class Leaf:
    def __init__(self, label):
        self.label = label
    
    def __str__(self):
        return f"l: {self.label}"