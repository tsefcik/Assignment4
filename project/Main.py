from builtins import print
from project import Abalone as a
from project import Car as c
from project import Segmentation as s
from project import FiveFold as ff
import sys

"""
This is the main driver class for Assignment#3.  @author: Tyler Sefcik
"""


class Main:

    # Segmentation setup and run
    def run_seg(self, filename, column_names, sortby):
        # Setup data
        seg = s.Segmentation()
        seg_data = seg.setup_data(filename=filename, column_names=column_names)
        # Split the data set into 10% and 90%
        seg_validation_data = seg_data.sample(frac=.10)
        seg_data_rest = seg_data.drop(seg_validation_data.index)
        # Reset indexes on data frames
        seg_validation_data.reset_index(inplace=True)
        seg_data_rest.reset_index(inplace=True)

        print(seg_validation_data)
        print()
        print(seg_data_rest)

        # Setup five fold cross validation
        five_fold = ff.FiveFold()
        seg1, seg2, seg3, seg4, seg5 = five_fold.five_fold_sort_class(data=seg_data_rest, sortby=sortby)

        return seg_data

    # Abalone setup and run
    def run_abalone(self, filename, column_names, sortby):
        # Setup data
        abalone = a.Abalone()
        abalone_data = abalone.setup_data(filename=filename, column_names=column_names)
        # Split the data set into 10% and 90%
        abalone_validation_data = abalone_data.sample(frac=.10)
        abalone_data_rest = abalone_data.drop(abalone_validation_data.index)
        # Reset indexes on data frames
        abalone_validation_data.reset_index(inplace=True)
        abalone_data_rest.reset_index(inplace=True)

        print(abalone_validation_data)
        print()
        print(abalone_data_rest)

        # Setup five fold cross validation
        five_fold = ff.FiveFold()
        abalone1, abalone2, abalone3, abalone4, abalone5 = five_fold.five_fold_sort_class(data=abalone_data,
                                                                                          sortby=sortby)

        return abalone_data

    # Car setup and run
    def run_car(self, filename, column_names, sortby):
        # Setup data
        car = c.Car()
        car_data = car.setup_data(filename=filename, column_names=column_names)
        # Split the data set into 10% and 90%
        car_validation_data = car_data.sample(frac=.10)
        car_data_rest = car_data.drop(car_validation_data.index)
        # Reset indexes on data frames
        car_validation_data.reset_index(inplace=True)
        car_data_rest.reset_index(inplace=True)

        print(car_validation_data)
        print()
        print(car_data_rest)

        # Setup five fold cross validation
        five_fold = ff.FiveFold()
        car1, car2, car3, car4, car5 = five_fold.five_fold_sort_class(data=car_data, sortby=sortby)

        return car_data

    # Main driver to run all algorithms on each dataset
    def main(self):
        # Print all output to file
        # Comment out for printing in console
        # sys.stdout = open("./Assignment4Output.txt", "w")

        ##### Segmentation #####
        seg_names = ["class", "cen col", "cen row", "pix count", "sld -5", "sld -2", "vedge mean", "vedge sd",
                     "hedge mean", "hedge sd", "intensity mean", "rawred mean", "rawblue mean", "rawgreen mean",
                     "exred mean", "exblue mean", "exgreen mean", "value mean", "sat mean", "hue mean"]
        seg_data = self.run_seg(filename="data/segmentation.data", column_names=seg_names, sortby="class")

        print("Segmentation Data")
        print(seg_data)
        print()

        ##### Abalone #####
        abalone_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                         "shell weight", "rings"]
        abalone_data = self.run_abalone(filename="data/abalone.data", column_names=abalone_names, sortby="rings")

        print("Abalone Data")
        print(abalone_data)
        print()

        ##### Car #####
        car_names = ["buying", "maint", "doors", "persons", "lug boot", "safety", "class"]
        car_data = self.run_car(filename="data/car.data", column_names=car_names, sortby="class")

        print("Car Data")
        print(car_data)
        print()


if __name__ == "__main__":
    main = Main()
    main.main()
