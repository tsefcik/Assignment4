"""
This class represents a node in the decision tree.  It holds data, a feature it is connected to, and whether it is a
leaf node or not.  It also creates an empty dict for children nodes.
"""


class Node:
    def __init__(self, data, feature, leaf_node):
        self.data = data
        self.feature = feature
        self.leaf_node = leaf_node
        self.children = {}

