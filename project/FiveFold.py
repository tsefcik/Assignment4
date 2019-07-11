import pandas as pd

""" This class performs five fold cross validation on each dataset """


class FiveFold:

    def five_fold_sort_class(self, data, sortby):
        new_data = data.sort_values(by=sortby)

        group1 = pd.DataFrame(columns=new_data.columns)
        group2 = pd.DataFrame(columns=new_data.columns)
        group3 = pd.DataFrame(columns=new_data.columns)
        group4 = pd.DataFrame(columns=new_data.columns)
        group5 = pd.DataFrame(columns=new_data.columns)

        for index in range(len(new_data)):
            if index == 0:
                group1 = group1.append(new_data.loc[index], ignore_index=True)
            if index == 1:
                group2 = group2.append(new_data.loc[index], ignore_index=True)
            if index == 2:
                group3 = group3.append(new_data.loc[index], ignore_index=True)
            if index == 3:
                group4 = group4.append(new_data.loc[index], ignore_index=True)
            if index == 4:
                group5 = group5.append(new_data.loc[index], ignore_index=True)
            if index % 5 == 0:
                group1 = group1.append(new_data.loc[index], ignore_index=True)
            if index % 5 == 1:
                group2 = group2.append(new_data.loc[index], ignore_index=True)
            if index % 5 == 2:
                group3 = group3.append(new_data.loc[index], ignore_index=True)
            if index % 5 == 3:
                group4 = group4.append(new_data.loc[index], ignore_index=True)
            if index % 5 == 4:
                group5 = group5.append(new_data.loc[index], ignore_index=True)

        return group1, group2, group3, group4, group5
