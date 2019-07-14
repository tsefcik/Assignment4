from project import Node as node
from project import Util as util
import pandas as pd

"""
This class represents the Decision Tree that will be built using the ID3 Machine Learning algorithm.  It takes three 
parameters: data = data set, features = list of features in the data set, target_class = name of target class. 
"""


class DecisionTree:

    def create_decision_tree(self, data, features_list):
        class_list_from_data = []

        for index, row in data.iterrows():
            class_list_from_data.append(data.iloc[index, data.shape[1] - 1])

        unique_classes = set(class_list_from_data)

        if len(unique_classes) == 1 or len(features_list) == 1:
            new_node = node.Node(data=data, feature=-1, leaf_node=True)
            return new_node

        best_feature_available = util.select_best_feature_by_information_gain(data=data)

        features_list_original = features_list
        features_list.remove(best_feature_available)

        new_node = node.Node(data=data, feature=best_feature_available, leaf_node=False)

        best_feature_available_values = []

        for index, row in data.iterrows():
            best_feature_available_values.append(data.iloc[index].loc[best_feature_available])

        unique_best_feature_values = set(best_feature_available_values)

        for value in unique_best_feature_values:
            value_data = pd.DataFrame(columns=features_list_original)
            value_data = value_data.append(data[data[best_feature_available] == value], ignore_index=True)
            value_data.drop(best_feature_available, axis=1, inplace=True)
            class_col = value_data.pop("class")
            value_data["class"] = class_col
            print(value_data)
            new_node.children[value] = self.create_decision_tree(data=value_data, features_list=features_list)

        return new_node

    def get_tree_as_string(self, node2):
        if node2.leaf_node:
            string = '{' + str(node2.feature) + '}'
            return string
        else:
            string = ' { ' + str(node2.feature) + ': '
            for child_key in node2.children:
                string = string + self.get_tree_as_string(node2.children[child_key])
            string = string + ' } '
        return string
