import sys
import os
import pandas as pd


class ClearAllData(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Model_Train_Data_Files/"
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        
        self.issue_predicted_data = self.path + 'bert_predictions_with_issue_types.csv' # Data with issues prediction
        self.save_with_reason = self.path + "bert_predictions_issues_with_reason.csv"  # Data file with possible reasons behind predicted issues.
    
    def clear_data(self):
        df_issue_predicted = pd.read_csv(self.issue_predicted_data)
        df_save_with_reason = pd.read_csv(self.save_with_reason)

        # Keep only the header
        df_issue_predicted = df_issue_predicted.iloc[0:0]
        df_save_with_reason = df_save_with_reason.iloc[0:0]

        # Save back to the same CSV files
        df_issue_predicted.to_csv(self.issue_predicted_data, index=False)
        df_save_with_reason.to_csv(self.save_with_reason, index=False)
    
    
if __name__ == "__main__":
    obj = ClearAllData()
    obj.clear_data()
