import pandas as pd

"""
    This class is used to setup the Abalone data that will be used for processing.
"""


class Abalone:

    def setup_data(self, filename, column_names):
        # Read in data file and turn into data structure

        data = pd.read_csv(filename,
                           sep=",",
                           header=0,
                           names=column_names)

        return data
