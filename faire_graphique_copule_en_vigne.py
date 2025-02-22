# Adapté à partir du package pyvinecopulib
import math
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def get_name(vc, tree, edge, vars_names=None):
    m = vc.matrix
    d = m.shape[0]  # Number of rows (equivalent to nrow(M) in R)

    # Conditioned set
    bef_indices = [d - edge - 1, tree]  # Adjusted for zero-indexing
    bef = ",".join([vars_names[int(m[i, edge]) - 1] for i in bef_indices])

    # Conditioning set
    if tree > 0:
        aft = ",".join(
            [vars_names[int(m[i - 1, edge]) - 1] for i in range(tree, 0, -1)]
        )
    else:
        aft = ""

    # Separator
    sep = ";" if tree > 0 else ""

    # Combine everything
    return bef + sep + aft


def get_graph(tree, vc, vars_names):
    m = vc.matrix
    d = vc.dim

    adj_mat = np.zeros((d - tree, d - tree), dtype=int)

    vars_names = ["inter- national" if item == "international" else item for item in vars_names]
    vars_names = ["house- hold" if item == "household" else item for item in vars_names]
    vars_names = ["beve- rages" if item == "beverages" else item for item in vars_names]

    # Extract node and edge labels as numbers
    if tree > 0:
        vertices = np.zeros((d - tree, tree + 1), dtype=int)
        for j in range(d - tree):
            rows = np.array([d - j - 1, *range(tree - 1, -1, -1)])
            vertices[j, :] = m[rows, j]
    else:
        vertices = np.diag(m[d - 1:: -1, :]).reshape(-1, 1)

    edges = np.zeros((d - tree - 1, tree + 2), dtype=int)
    for j in range(d - tree - 1):
        rows = np.array([d - j - 1, *range(tree, -1, -1)])
        edges[j, :] = m[rows, j]

    # Build adjacency matrix by matching vertices and edges
    edge_labels = {}
    for i in range(edges.shape[0]):
        ind_i = []
        for j in range(vertices.shape[0]):
            if np.all(np.isin(vertices[j, :], edges[i, :])):
                ind_i.append(j)
        adj_mat[ind_i[0], ind_i[1]] = 1
        adj_mat[ind_i[1], ind_i[0]] = 1
        edge_labels[(ind_i[0], ind_i[1])] = get_name(vc, tree, i, vars_names)

    # Node labels
    if tree > 0:
        node_labels = {
            i: get_name(vc, tree - 1, i, vars_names) for i in range(d - tree)
        }
    else:
        node_labels = {j: vars_names[int(m[d - j - 1, j]) - 1] for j in range(d)}

    return adj_mat, node_labels, edge_labels


def vinecop_plot(
        cop,
        tree: List[int] = None,
        add_edge_labels=True,
        layout: str = "graphviz",
        vars_names: List[str] = None,
) -> None:
    if not tree:
        if cop.trunc_lvl > 8:
            raise ValueError(
                "The dimension and truncation level are too high to visualize all trees, "
                "please specify a list of trees to visualize."
            )
        else:
            tree = list(range(cop.trunc_lvl))

    if "spring_layout" in layout:  # change to spring_layout if needed
        layout = "spring_layout"

    if vars_names is not None:
        if len(vars_names) != cop.dim:
            raise ValueError(
                "The number of variable names must be equal to the dimension of the vine copula."
            )
    else:
        vars_names = [str(i) for i in range(cop.dim)]

    mat = cop.matrix
    if len(tree) > 3:
        n_col = 2
    else:
        n_col = 1
    n_row = math.ceil(len(tree) / n_col)

    fig, ax = plt.subplots(
        n_row, n_col, figsize=(5, 8)
    )  # fits quite ok up to 8 nodes
    if n_row == 1 and n_col == 1:
        ax = np.array([[ax]])  # Wrap it in a 2D array
    elif n_row == 1:
        ax = np.array([ax])  # Wrap the 1D row into a 2D array
    elif n_col == 1:
        ax = np.array(
            [[ax[i]] for i in range(n_row)]
        )  # Convert the 1D column into a 2D array
    else:
        pass

    col = 0  # initialisation
    row = 0  # initialisation
    for t in tree:
        ax[row, col].set_title("Tree {}".format(t))
        ax[row, col].margins(0.2)
        adj_mat, node_labels, edge_labels = get_graph(t, cop, vars_names)
        g = nx.from_numpy_array(adj_mat)
        # try to avoid crossing edges, there is no straight solution for it,
        # there are other options as well, but this works fine
        # There is also an implementation of graphviz:
        g.graph["graph"] = {"rankdir": "LR"}  # also possible 'TB', 'BT' and 'RL'
        if "graphviz" in layout:
            pos = nx.drawing.nx_pydot.graphviz_layout(g, prog="dot")
        else:
            pos = nx.spring_layout(g)

        # Default parameters for nx.draw
        node_labels = {k: "\n".join(v.split()) for k, v in node_labels.items()}
        nodes_defaults = {
            "with_labels": True,
            "labels": node_labels,
            "node_size": 550,  # np.max([200 + 20 * len(name) for name in vars_names]),
            "node_color": "lightblue",
            "edge_color": "gray",
            "font_size": 5,
        }

        nx.draw(
            g, pos=pos, ax=ax[row, col], **nodes_defaults
        )  # actual drawing of the network in the subplot

        if add_edge_labels:
            # Default parameters for nx.draw_networkx_edge_labels
            edge_defaults = {
                "edge_labels": edge_labels,
                "font_size": 6,
                "font_color": "black",
            }

            nx.draw_networkx_edge_labels(g, pos=pos, ax=ax[row, col], **edge_defaults)

        row += 1
        if row >= n_row:
            col += 1
            row = 0
        if len(mat[0]) - 1 % n_row > 0:
            ax[-1, -1].axis("off")

    plt.tight_layout()  # fit the graph within the figsize nicely
    return fig
