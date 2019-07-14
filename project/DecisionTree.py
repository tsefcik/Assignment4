from project import Node as node
from project import Util as util
import pandas as pd

"""
This class represents the Decision Tree that will be built using the ID3 Machine Learning algorithm.  It takes two 
parameters: data = data set and features = list of features in the data set. 
"""


class DecisionTree:
    def __init__(self):
        self.prediction = None

    def create_decision_tree(self, data, features_list):
        # Initialize a list to keep the class values in
        class_list_from_data = []

        # Add all class values to the above list
        for index, row in data.iterrows():
            class_list_from_data.append(data.iloc[index, data.shape[1] - 1])

        # Get unique classes (returns the different types of classes in this particular data set
        unique_classes = set(class_list_from_data)

        # We know we have a leaf node if there is only one class value in the class list
        # We know we have a leaf node if there is only one feature left in the feature list
        if len(unique_classes) == 1 or len(features_list) == 2:
            most_common_class = util.most_common_class(class_list_from_data)
            # Create leaf node since there will not be any children attached at this point
            # -99 represents a leaf node since no feature is attached
            new_node = node.Node(data=data, feature=most_common_class, leaf_node=True)
            self.prediction = most_common_class
            return new_node

        # Get best feature left in this data
        best_feature_available = util.select_best_feature_by_information_gain(data=data)

        # Make a copy of the original feature list before removing it from the feature list since we will need it below
        features_list_original = features_list
        # Remove best available feature since we will not be using it again
        features_list.remove(best_feature_available)

        # Create new node that will have children since it is not a leaf node
        new_node = node.Node(data=data, feature=best_feature_available, leaf_node=False)

        best_feature_available_values = []

        for index, row in data.iterrows():
            best_feature_available_values.append(data.iloc[index].loc[best_feature_available])

        unique_best_feature_values = set(best_feature_available_values)

        for value in unique_best_feature_values:
            value_data = pd.DataFrame(columns=features_list_original)
            value_data = value_data.append(data[data[best_feature_available] == value], ignore_index=True, sort=True)
            value_data.drop(best_feature_available, axis=1, inplace=True)
            class_col = value_data.pop("class")
            value_data["class"] = class_col
            new_node.children[value] = self.create_decision_tree(data=value_data, features_list=features_list)

        return new_node

    # Run test with data set to predict accuracy
    def run_test(self, data, tree):
        # Keep list of predictions
        predictions = []
        # Initialize a list to keep the class values in
        class_list_from_data = []

        # Add all class values to the above list
        for index, row in data.iterrows():
            class_list_from_data.append(data.iloc[index, data.shape[1] - 1])
            # Set current node to variable
            curr_node = tree

            # Loop until we hit a leaf node
            while not curr_node.leaf_node:
                # Get feature value at current row/feature
                feature_val = data.iloc[index].loc[curr_node.feature]

                # Set children variable with current nodes children since we need to look through them
                curr_node_children = curr_node.children
                # If the feature value is not a part of the current node in some way, break out of the loop, we hit a
                # leaf node
                if feature_val not in curr_node_children:
                    break

                # Set current node to child node where the feature matches
                curr_node = curr_node.children[feature_val]
                self.prediction = curr_node.feature

            # Get prediction by most popular class from that node to be used for comparison later
            prediction = self.prediction
            predictions.append(prediction)

        # Compare the actual classes to the predicted classes found by iterating through the tree
        total_correct = 0
        for index in range(0, len(predictions) - 1):
            if predictions[index] == class_list_from_data[index]:
                total_correct += 1

        # Accuracy of the decision tree on all rows in data
        accuracy = (total_correct/len(predictions))*100

        return accuracy

    # Prune Tree
    def prune_tree(self, data, tree):
        # Consider pruning as long as we are not on a leaf node
        if not tree.leaf_node:
            # Iterate through children nodes and recursively call prune_tree (will stop once we hit a leaf node)
            for value, child in tree.children.items():
                self.prune_tree(data, child)

            # Get the original score
            original_score = self.run_test(data, tree)

            # Make this node a leaf node and get rid of children if there were some before this
            tree.leaf_node = True
            tree_children = tree.children
            tree.children = {}
            # Get updated score
            updated_score = self.run_test(data, tree)

            # Compare score
            if original_score > updated_score:
                tree.leaf_node = False
                tree.children = tree_children
            return tree
        else:
            return tree
