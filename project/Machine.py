import pandas as pd
from project import SetupData as sd

"""
    This class is used to setup the Machine data that will be used for processing.
"""


class Machine:

    def setup_data(self, filename, column_names, columns_to_drop):
        # Read in data file and turn into data structure

        data = pd.read_csv(filename,
                           sep=",",
                           header=0,
                           names=column_names)

        # Move first column to last to make it easier to work with
        first_col = data.iloc[:, 0]
        data = data.iloc[:, 1:]
        data = data.join(first_col)

        # Drop columns that we don't want to consider
        for name in columns_to_drop:
            data = data.drop(name, 1)

        # Normalize data on columns with appropriate values
        data = data.iloc[:, 0:data.shape[1] - 1]
        setup = sd.SetupData()
        normalized = setup.normalize_data(data=data)
        normalized = normalized.join(first_col)

        return normalized
