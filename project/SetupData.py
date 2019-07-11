import pandas as pd
import numpy as np
from sklearn import preprocessing


"""
    This class is used to normalize the datasets that will be used for processing.
"""


class SetupData:

    """
        This method loads the data file into normalized data.
    """

    def normalize_data(self, data):
        data = data.astype(np.float)
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        # Normalize data
        normalized_data = scaler.fit_transform(data)
        # Put normalized_data into dataframe
        normalized_dataframe = pd.DataFrame(data=normalized_data)

        return normalized_dataframe

    # This method returns a numerical value for each month
    def change_month(self, month):
        month_change = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12
        }

        month_num = month_change.get(month)

        return month_num

    # This method returns a numerical value for each day
    def change_day(self, day):
        day_change = {
            "sun": 1,
            "mon": 2,
            "tue": 3,
            "wed": 4,
            "thu": 5,
            "fri": 6,
            "sat": 7
        }

        day_num = day_change.get(day)

        return day_num
