import pandas as pd
from project import SetupData as sd

"""
    This class is used to setup the Car data that will be used for processing.
"""


class Car:

    def setup_data(self, filename, column_names):
        # Read in data file and turn into data structure

        data = pd.read_csv(filename,
                           sep=",",
                           header=0,
                           names=column_names)

        return data
