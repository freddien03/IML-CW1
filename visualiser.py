import matplotlib.pyplot as plt
from node import Node, Leaf
from decision_tree import Decision_Tree_Classifier
import numpy as np

def visualise_tree(ax, tree):
  visualise_decision(ax, tree, 1, 0, 20)

def visualise_decision(ax, node, depth, posx, horizontal_scale=20):
  posy = -depth*1.5
  child_posy = -(depth+1)*1.5

  node_text = f"X{node.attr} < {round(node.val, 2)}"

  left_posx = posx - horizontal_scale/(2**depth)
  right_posx = posx + horizontal_scale/(2**depth)

  box_style = dict(boxstyle="round,pad=0.3", facecolor='lightblue')
  ax.text(posx, posy, node_text, 
               ha='center', va='center',
               bbox=box_style, fontsize=horizontal_scale/(depth))
  ax.plot([posx, left_posx], [posy, child_posy], 'k-')
  ax.plot([posx, right_posx], [posy, child_posy], 'k-')

  if isinstance(node.left, Node):
    visualise_decision(ax, node.left, depth+1, left_posx, horizontal_scale)
  else:
    visualise_leaf(ax, node.left, depth+1, left_posx, horizontal_scale)
  
  if isinstance(node.right, Node):
    visualise_decision(ax, node.right, depth+1, right_posx, horizontal_scale)
  else:
    visualise_leaf(ax, node.right, depth+1, right_posx, horizontal_scale)


def visualise_leaf(ax, node, depth, posx, horizontal_scale):
  posy = -depth*1.5
  node_text = node.label
  box_style = dict(boxstyle="round,pad=0.3", facecolor='lightgreen')
  ax.text(posx, posy, node_text, 
               ha='center', va='center',
               bbox=box_style, fontsize=horizontal_scale/(depth-1))

def main():
  nodel = Node(2, 10, Leaf(-1), Leaf(-2))
  noder = Node(4, 24, Leaf(-3), Leaf(-1))
  tree = Node(1, 5, nodel, noder)

  clean = np.loadtxt("wifi_db/clean_dataset.txt")
  noisy = np.loadtxt("wifi_db/noisy_dataset.txt")
  model = Decision_Tree_Classifier()
  tree, depth = model.decision_tree_learning(clean, 0)

  fig, ax = plt.subplots()
  visualise_tree(ax, tree)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()