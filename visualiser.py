import matplotlib.pyplot as plt
from node import Leaf

def visualise_tree(ax, tree, hsep=1.6, vstep=1.6,
                   node_fs=10, leaf_fs=10, box_pad=0.3, decimals=2):
    # draws the decision tree with given layout params
    ax.axis("off")
    pos = _layout_positions(tree, hsep=hsep, vstep=vstep)
    _draw(ax, tree, pos, node_fs=node_fs, leaf_fs=leaf_fs,
          box_pad=box_pad, decimals=decimals)

    xs = [x for (x, _) in pos.values()]
    ys = [y for (_, y) in pos.values()]
    pad_x = max(1.0, hsep)
    pad_y = max(0.5, vstep)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.margins(x=0.02, y=0.02)

def save_tree_png(tree, path="clean_tree.png", figsize=(14, 7),
                  hsep=1.6, vstep=1.6, show=False,
                  node_fs=10, leaf_fs=10, box_pad=0.3, decimals=2, dpi=150):
    # draws tree and saves a PNG
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    visualise_tree(ax, tree, hsep=hsep, vstep=vstep,
                   node_fs=node_fs, leaf_fs=leaf_fs,
                   box_pad=box_pad, decimals=decimals)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def _layout_positions(tree, hsep, vstep):
    # returns dict of object ids to coords using inorder leaf columns
    pos = {}
    xcur = [0.0]
    def dfs(node, depth):
        if isinstance(node, Leaf):
            x = xcur[0] * hsep; xcur[0] += 1.0
            pos[id(node)] = (x, -depth * vstep)
            return x
        xl = dfs(node.left, depth + 1)
        xr = dfs(node.right, depth + 1)
        x = (xl + xr) / 2.0
        pos[id(node)] = (x, -depth * vstep)
        return x
    dfs(tree, 1)
    return pos

def _draw(ax, node, pos, node_fs, leaf_fs, box_pad, decimals):
    if isinstance(node, Leaf):
        _draw_leaf(ax, node, pos, leaf_fs, box_pad)
        return
    x, y = pos[id(node)]
    xl, yl = pos[id(node.left)]
    xr, yr = pos[id(node.right)]

    box = dict(boxstyle=f"round,pad={box_pad}", facecolor="lightblue")
    text = f"X{node.attr} ≤ {format(node.val, f'.{decimals}f')}"
    ax.text(x, y, text, ha="center", va="center", bbox=box, fontsize=node_fs)

    ax.plot([x, xl], [y, yl], "k-", linewidth=1.0)
    ax.plot([x, xr], [y, yr], "k-", linewidth=1.0)

    _draw(ax, node.left, pos, node_fs, leaf_fs, box_pad, decimals)
    _draw(ax, node.right, pos, node_fs, leaf_fs, box_pad, decimals)

def _draw_leaf(ax, node, pos, leaf_fs, box_pad):
    x, y = pos[id(node)]
    box = dict(boxstyle=f"round,pad={box_pad}", facecolor="lightgreen")
    text = str(int(node.label)) if float(node.label).is_integer() else str(node.label)
    ax.text(x, y, text, ha="center", va="center", bbox=box, fontsize=leaf_fs)
