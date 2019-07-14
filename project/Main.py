from builtins import print
from project import Abalone as a
from project import Car as c
from project import Segmentation as s
from project import FiveFold as ff
from project import DecisionTree as dt
import numpy as np
import sys

"""
This is the main driver class for Assignment#4.  @author: Tyler Sefcik
"""


class Main:

    # Segmentation setup and run
    def run_seg(self, filename, sortby):
        print()
        seg_names = ["class", "cen col", "cen row", "pix count", "sld -5", "sld -2", "vedge mean", "vedge sd",
                     "hedge mean", "hedge sd", "intensity mean", "rawred mean", "rawblue mean", "rawgreen mean",
                     "exred mean", "exblue mean", "exgreen mean", "value mean", "sat mean", "hue mean"]
        # Setup data
        seg = s.Segmentation()
        seg_data = seg.setup_data(filename=filename, column_names=seg_names)
        # Split the data set into 10% and 90%
        seg_validation_data = seg_data.sample(frac=.10)
        seg_data_rest = seg_data.drop(seg_validation_data.index)
        # Reset indexes on data frames
        seg_validation_data.reset_index(inplace=True)
        seg_data_rest.reset_index(inplace=True)

        # Setup five fold cross validation
        five_fold = ff.FiveFold()
        seg1, seg2, seg3, seg4, seg5 = five_fold.five_fold_sort_class(data=seg_data_rest, sortby=sortby)
        seg1.drop(columns='index', axis=1, inplace=True)
        seg2.drop(columns='index', axis=1, inplace=True)
        seg3.drop(columns='index', axis=1, inplace=True)
        seg4.drop(columns='index', axis=1, inplace=True)
        seg5.drop(columns='index', axis=1, inplace=True)

        tree1 = dt.DecisionTree()
        tree_node1 = tree1.create_decision_tree(data=seg1, features_list=seg_names)
        accuracy1 = tree1.run_test(seg_validation_data, tree_node1)
        print("Unpruned accuracy for seg1: " + str(accuracy1) + "%")
        tree_pruned1 = tree1.prune_tree(seg_validation_data, tree_node1)
        accuracy_pruned1 = tree1.run_test(seg_validation_data, tree_pruned1)
        print("Pruned accuracy for seg1: " + str(accuracy_pruned1) + "%")
        print()

        seg_names = ["class", "cen col", "cen row", "pix count", "sld -5", "sld -2", "vedge mean", "vedge sd",
                     "hedge mean", "hedge sd", "intensity mean", "rawred mean", "rawblue mean", "rawgreen mean",
                     "exred mean", "exblue mean", "exgreen mean", "value mean", "sat mean", "hue mean"]
        tree2 = dt.DecisionTree()
        tree_node2 = tree2.create_decision_tree(data=seg2, features_list=seg_names)
        accuracy2 = tree2.run_test(seg_validation_data, tree_node2)
        print("Unpruned accuracy for seg2: " + str(accuracy2) + "%")
        tree_pruned2 = tree2.prune_tree(seg_validation_data, tree_node2)
        accuracy_pruned2 = tree2.run_test(seg_validation_data, tree_pruned2)
        print("Pruned accuracy for seg2: " + str(accuracy_pruned2) + "%")
        print()

        seg_names = ["class", "cen col", "cen row", "pix count", "sld -5", "sld -2", "vedge mean", "vedge sd",
                     "hedge mean", "hedge sd", "intensity mean", "rawred mean", "rawblue mean", "rawgreen mean",
                     "exred mean", "exblue mean", "exgreen mean", "value mean", "sat mean", "hue mean"]
        tree3 = dt.DecisionTree()
        tree_node3 = tree3.create_decision_tree(data=seg3, features_list=seg_names)
        accuracy3 = tree3.run_test(seg_validation_data, tree_node3)
        print("Unpruned accuracy for seg3: " + str(accuracy3) + "%")
        tree_pruned3 = tree3.prune_tree(seg_validation_data, tree_node3)
        accuracy_pruned3 = tree3.run_test(seg_validation_data, tree_pruned3)
        print("Pruned accuracy for seg3: " + str(accuracy_pruned3) + "%")
        print()

        seg_names = ["class", "cen col", "cen row", "pix count", "sld -5", "sld -2", "vedge mean", "vedge sd",
                     "hedge mean", "hedge sd", "intensity mean", "rawred mean", "rawblue mean", "rawgreen mean",
                     "exred mean", "exblue mean", "exgreen mean", "value mean", "sat mean", "hue mean"]
        tree4 = dt.DecisionTree()
        tree_node4 = tree4.create_decision_tree(data=seg4, features_list=seg_names)
        accuracy4 = tree4.run_test(seg_validation_data, tree_node4)
        print("Unpruned accuracy for seg4: " + str(accuracy4) + "%")
        tree_pruned4 = tree4.prune_tree(seg_validation_data, tree_node4)
        accuracy_pruned4 = tree4.run_test(seg_validation_data, tree_pruned4)
        print("Pruned accuracy for seg4: " + str(accuracy_pruned4) + "%")
        print()

        seg_names = ["class", "cen col", "cen row", "pix count", "sld -5", "sld -2", "vedge mean", "vedge sd",
                     "hedge mean", "hedge sd", "intensity mean", "rawred mean", "rawblue mean", "rawgreen mean",
                     "exred mean", "exblue mean", "exgreen mean", "value mean", "sat mean", "hue mean"]
        tree5 = dt.DecisionTree()
        tree_node5 = tree5.create_decision_tree(data=seg5, features_list=seg_names)
        accuracy5 = tree5.run_test(seg_validation_data, tree_node5)
        print("Unpruned accuracy for seg5: " + str(accuracy5) + "%")
        tree_pruned5 = tree5.prune_tree(seg_validation_data, tree_node5)
        accuracy_pruned5 = tree5.run_test(seg_validation_data, tree_pruned5)
        print("Pruned accuracy for seg5: " + str(accuracy_pruned5) + "%")
        print()

        unpruned_accuracy_average = np.average([accuracy1, accuracy2, accuracy3, accuracy4, accuracy5])
        pruned_accuracy_average = np.average([accuracy_pruned1, accuracy_pruned2, accuracy_pruned3, accuracy_pruned4,
                                             accuracy_pruned5])

        print("Unpruned accuracy average for seg data: " + str(unpruned_accuracy_average) + "%")
        print("Pruned accuracy average for seg data: " + str(pruned_accuracy_average) + "%")
        print()

    # Abalone setup and run
    def run_abalone(self, filename, sortby):
        print()
        abalone_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                         "shell weight", "class"]
        # Setup data
        abalone = a.Abalone()
        abalone_data = abalone.setup_data(filename=filename, column_names=abalone_names)
        # Split the data set into 10% and 90%
        abalone_validation_data = abalone_data.sample(frac=.10)
        abalone_data_rest = abalone_data.drop(abalone_validation_data.index)
        # Reset indexes on data frames
        abalone_validation_data.reset_index(inplace=True)
        abalone_data_rest.reset_index(inplace=True)

        # Setup five fold cross validation
        five_fold = ff.FiveFold()
        abalone1, abalone2, abalone3, abalone4, abalone5 = five_fold.five_fold_sort_class(data=abalone_data,
                                                                                          sortby=sortby)

        tree1 = dt.DecisionTree()
        tree_node1 = tree1.create_decision_tree(data=abalone1, features_list=abalone_names)
        accuracy1 = tree1.run_test(abalone_validation_data, tree_node1)
        print("Unpruned accuracy for abalone1: " + str(accuracy1) + "%")
        tree_pruned1 = tree1.prune_tree(abalone_validation_data, tree_node1)
        accuracy_pruned1 = tree1.run_test(abalone_validation_data, tree_pruned1)
        print("Pruned accuracy for abalone1: " + str(accuracy_pruned1) + "%")
        print()

        abalone_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                         "shell weight", "class"]
        tree2 = dt.DecisionTree()
        tree_node2 = tree2.create_decision_tree(data=abalone2, features_list=abalone_names)
        accuracy2 = tree2.run_test(abalone_validation_data, tree_node2)
        print("Unpruned accuracy for abalone2: " + str(accuracy2) + "%")
        tree_pruned2 = tree2.prune_tree(abalone_validation_data, tree_node2)
        accuracy_pruned2 = tree2.run_test(abalone_validation_data, tree_pruned2)
        print("Pruned accuracy for abalone2: " + str(accuracy_pruned2) + "%")
        print()

        abalone_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                         "shell weight", "class"]
        tree3 = dt.DecisionTree()
        tree_node3 = tree3.create_decision_tree(data=abalone3, features_list=abalone_names)
        accuracy3 = tree3.run_test(abalone_validation_data, tree_node3)
        print("Unpruned accuracy for abalone3: " + str(accuracy3) + "%")
        tree_pruned3 = tree3.prune_tree(abalone_validation_data, tree_node3)
        accuracy_pruned3 = tree3.run_test(abalone_validation_data, tree_pruned3)
        print("Pruned accuracy for abalone3: " + str(accuracy_pruned3) + "%")
        print()

        abalone_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                         "shell weight", "class"]
        tree4 = dt.DecisionTree()
        tree_node4 = tree4.create_decision_tree(data=abalone4, features_list=abalone_names)
        accuracy4 = tree4.run_test(abalone_validation_data, tree_node4)
        print("Unpruned accuracy for abalone4: " + str(accuracy4) + "%")
        tree_pruned4 = tree4.prune_tree(abalone_validation_data, tree_node4)
        accuracy_pruned4 = tree4.run_test(abalone_validation_data, tree_pruned4)
        print("Pruned accuracy for abalone4: " + str(accuracy_pruned4) + "%")
        print()

        abalone_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                         "shell weight", "class"]
        tree5 = dt.DecisionTree()
        tree_node5 = tree5.create_decision_tree(data=abalone5, features_list=abalone_names)
        accuracy5 = tree5.run_test(abalone_validation_data, tree_node5)
        print("Unpruned accuracy for abalone5: " + str(accuracy5) + "%")
        tree_pruned5 = tree5.prune_tree(abalone_validation_data, tree_node5)
        accuracy_pruned5 = tree5.run_test(abalone_validation_data, tree_pruned5)
        print("Pruned accuracy for abalone5: " + str(accuracy_pruned5) + "%")
        print()

        unpruned_accuracy_average = np.average([accuracy1, accuracy2, accuracy3, accuracy4, accuracy5])
        pruned_accuracy_average = np.average([accuracy_pruned1, accuracy_pruned2, accuracy_pruned3, accuracy_pruned4,
                                             accuracy_pruned5])

        print("Unpruned accuracy average for abalone data: " + str(unpruned_accuracy_average) + "%")
        print("Pruned accuracy average for abalone data: " + str(pruned_accuracy_average) + "%")
        print()

    # Car setup and run
    def run_car(self, filename, sortby):
        print()
        car_names = ["buying", "maint", "doors", "persons", "lug boot", "safety", "class"]
        # Setup data
        car = c.Car()
        car_data = car.setup_data(filename=filename, column_names=car_names)
        # Split the data set into 10% and 90%
        car_validation_data = car_data.sample(frac=.10)
        car_data_rest = car_data.drop(car_validation_data.index)
        # Reset indexes on data frames
        car_validation_data.reset_index(inplace=True)
        car_data_rest.reset_index(inplace=True)

        # print(car_validation_data)
        # print()
        # print(car_data_rest)

        # Setup five fold cross validation
        five_fold = ff.FiveFold()
        car1, car2, car3, car4, car5 = five_fold.five_fold_sort_class(data=car_data_rest, sortby=sortby)
        car1.drop(columns='index', axis=1, inplace=True)
        car2.drop(columns='index', axis=1, inplace=True)
        car3.drop(columns='index', axis=1, inplace=True)
        car4.drop(columns='index', axis=1, inplace=True)
        car5.drop(columns='index', axis=1, inplace=True)

        tree1 = dt.DecisionTree()
        tree_node1 = tree1.create_decision_tree(data=car1, features_list=car_names)
        accuracy1 = tree1.run_test(car_validation_data, tree_node1)
        print("Unpruned accuracy for car1: " + str(accuracy1) + "%")
        tree_pruned1 = tree1.prune_tree(car_validation_data, tree_node1)
        accuracy_pruned1 = tree1.run_test(car_validation_data, tree_pruned1)
        print("Pruned accuracy for car1: " + str(accuracy_pruned1) + "%")
        print()

        car_names = ["buying", "maint", "doors", "persons", "lug boot", "safety", "class"]
        tree2 = dt.DecisionTree()
        tree_node2 = tree2.create_decision_tree(data=car2, features_list=car_names)
        accuracy2 = tree2.run_test(car_validation_data, tree_node2)
        print("Unpruned accuracy for car2: " + str(accuracy2) + "%")
        tree_pruned2 = tree2.prune_tree(car_validation_data, tree_node2)
        accuracy_pruned2 = tree2.run_test(car_validation_data, tree_pruned2)
        print("Pruned accuracy for car2: " + str(accuracy_pruned2) + "%")
        print()

        car_names = ["buying", "maint", "doors", "persons", "lug boot", "safety", "class"]
        tree3 = dt.DecisionTree()
        tree_node3 = tree3.create_decision_tree(data=car3, features_list=car_names)
        accuracy3 = tree3.run_test(car_validation_data, tree_node3)
        print("Unpruned accuracy for car3: " + str(accuracy3) + "%")
        tree_pruned3 = tree3.prune_tree(car_validation_data, tree_node3)
        accuracy_pruned3 = tree3.run_test(car_validation_data, tree_pruned3)
        print("Pruned accuracy for car3: " + str(accuracy_pruned3) + "%")
        print()

        car_names = ["buying", "maint", "doors", "persons", "lug boot", "safety", "class"]
        tree4 = dt.DecisionTree()
        tree_node4 = tree4.create_decision_tree(data=car4, features_list=car_names)
        accuracy4 = tree4.run_test(car_validation_data, tree_node4)
        print("Unpruned accuracy for car4: " + str(accuracy4) + "%")
        tree_pruned4 = tree4.prune_tree(car_validation_data, tree_node4)
        accuracy_pruned4 = tree4.run_test(car_validation_data, tree_pruned4)
        print("Pruned accuracy for car4: " + str(accuracy_pruned4) + "%")
        print()

        car_names = ["buying", "maint", "doors", "persons", "lug boot", "safety", "class"]
        tree5 = dt.DecisionTree()
        tree_node5 = tree5.create_decision_tree(data=car5, features_list=car_names)
        accuracy5 = tree5.run_test(car_validation_data, tree_node5)
        print("Unpruned accuracy for car5: " + str(accuracy5) + "%")
        tree_pruned5 = tree5.prune_tree(car_validation_data, tree_node5)
        accuracy_pruned5 = tree5.run_test(car_validation_data, tree_pruned5)
        print("Pruned accuracy for car5: " + str(accuracy_pruned5) + "%")
        print()

        unpruned_accuracy_average = np.average([accuracy1, accuracy2, accuracy3, accuracy4, accuracy5])
        pruned_accuracy_average = np.average([accuracy_pruned1, accuracy_pruned2, accuracy_pruned3, accuracy_pruned4,
                                             accuracy_pruned5])

        print("Unpruned accuracy average for car data: " + str(unpruned_accuracy_average) + "%")
        print("Pruned accuracy average for car data: " + str(pruned_accuracy_average) + "%")
        print()

    # Main driver to run all algorithms on each dataset
    def main(self):
        # Print all output to file
        # Comment out for printing in console
        # sys.stdout = open("./Assignment4Output.txt", "w")

        # ##### Car #####
        # self.run_car(filename="data/car.data", sortby="class")
        #
        # ##### Segmentation #####
        # self.run_seg(filename="data/segmentation.data", sortby="class")

        ##### Abalone #####
        self.run_abalone(filename="data/abalone.data", sortby="class")


if __name__ == "__main__":
    main = Main()
    main.main()
