import numpy as np

"""
This class contains utility methods used for making calculations in the Decision Tree.
"""


# Calculate entropy for a given data set
def calculate_entropy(data):
    # Get length of the data being calculated
    num_rows = len(data)
    # Create blank label dictionary to store class labels and number of occurrences
    label_dict = {}
    # Initialize total entropy
    total_entropy = 0

    # Iterate through each row and pull the label, then add it to the label dictionary/keep track of number of
    # occurrences
    for row in data:
        single_label = row[-1]

        if single_label not in label_dict.keys():
            label_dict[single_label] = 0
        label_dict[single_label] += 1

    # For every key(label) in the label dictionary, get the probability, then use that to further calculate the total
    # entropy
    for key in label_dict:
        probability = label_dict[key]/num_rows
        total_entropy -= probability*np.log2(probability)

    return total_entropy


# Get the best feature to split the data on by calculating the information gain for a given a data set
def select_best_feature_by_information_gain(data):
    # Find number of features in for the given data set
    num_features = data.shape[1] - 1

    # Calculate the original entropy of the original data set
    original_entropy = calculate_entropy(data=data)

    # Initialize total information gain to be used to compare information gain on different features
    total_information_gain = 0

    # Initialize best feature variable to be used to choose the best feature after all iterations are done
    best_feature = None

    # Iterate through each feature
    for index in range(num_features):
        # Initialize a list to hold the values of different features
        feature_list = []

        # Add the values of the features to the list
        for index2 in range(len(data)):
            feature_list.append(data.iloc[index2, index])
        # Retrieve the unique features from the feature list
        feature_values = set(feature_list)

        # Initialize entropy value for specific feature
        feature_entropy = 0

        # Iterate through each value in the different type of feature values
        for unique_value in feature_values:
            # Split the data set based on the feature with the best information gain
            split_data = split_dataset_for_max_info_gain(data=data, feature_index=index, feature_value=unique_value)
            probability = len(split_data)/len(data)
            entropy = probability*calculate_entropy(split_data)
            feature_entropy += entropy

        # Calculate information gain for this feature
        calculated_info_gain = original_entropy - feature_entropy

        # Set best feature/total information gain as long as this one beats out the previous, if not then this feature
        # is not the best feature so far
        if calculated_info_gain > total_information_gain:
            total_information_gain = calculated_info_gain
            best_feature = index

    return best_feature


# Split the given data set by the feature that has the maximum information gain
def split_dataset_for_max_info_gain(data, feature_index, feature_value):
    # List to store the split data that we come up with
    split_data = []

    # Iterate through each row in the data set
    for index, row in data.iterrows():
        # If the value for a given feature matches, split the data at that feature
        if data.iloc[index, feature_index] == feature_value:
            smaller_data_set = list(row[:feature_index])
            smaller_data_set.extend(row[feature_index + 1:])
            split_data.append(smaller_data_set)

    return split_data
