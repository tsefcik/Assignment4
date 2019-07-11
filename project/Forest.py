import pandas as pd
from project import SetupData as sd

"""
    This class is used to setup the Forest data that will be used for processing.
"""


class Forest:

    def setup_data(self, filename, column_names):
        # Read in data file and turn into data structure

        data = pd.read_csv(filename,
                           sep=",",
                           header=0,
                           names=column_names)

        setup = sd.SetupData()
        for index, row in data.iterrows():
            month_num = setup.change_month(row[2])
            data.at[index, "month"] = month_num

            day_num = setup.change_day(row[3])
            data.at[index, "day"] = day_num

        # Normalize data on columns with appropriate values
        data = data.iloc[:, :]
        normalized = setup.normalize_data(data=data)

        return normalized
