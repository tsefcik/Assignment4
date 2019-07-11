import pandas as pd
from project import SetupData as sd

"""
    This class is used to setup the Segmentation data that will be used for processing.
"""


class Segmentation:

    def setup_data(self, filename, column_names):
        # Read in data file and turn into data structure

        data = pd.read_csv(filename,
                           sep=",",
                           header=0,
                           names=column_names)

        # Move first column to last to make it easier to work with
        first_col = data.iloc[:, 0]
        data = data.iloc[:, 1:]
        data = data.join(first_col)

        return data
