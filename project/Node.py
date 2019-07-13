"""
This class represents a node in the decision tree.  It holds a value, a reference to the next node, and a reference
to it's chile node.
"""


class Node:
    def __init__(self, data, feature_list, target_class):
        self.data = data
        self.feature_list = feature_list
        self.target_class = target_class
        self.child = None

