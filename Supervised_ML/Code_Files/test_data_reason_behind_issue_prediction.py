# Transformer Libraries
import ollama

# Dataset Libraries
import pandas as pd

# Other Libraries
import os
import time
import sys
import concurrent.futures

class ReasonBehindIssuePrediction(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'ml_predictions_with_issue_types.csv') # Data with issues predictions
            self.save_with_reason = self.path + "ml_predictions_issues_with_reason.csv" # Data file with possible reasons behind predicted issues.
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'ml_predictions_with_issue_types.csv') # Data with issues predictions
            self.save_with_reason = self.path + "ml_predictions_issues_with_reason.csv" # Data file with possible reasons behind predicted issues.
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'ml_predictions_with_issue_types.csv') # Data with issues predictions
            self.save_with_reason = self.path + "ml_predictions_issues_with_reason.csv" # Data file with possible reasons behind predicted issues.
    
    def fetch_reason(self, index, row):
        if pd.notna(row['Issues']) and row['Issues'] != '':
            review = row['Content']
            response = ollama.chat(model='llama3', messages=[{
                'role': 'user', 
                'content': f"could you tell me if there is any issue with this customer review: '{review}'. Keep the answer not more than 20 words. Try to inspect carefully and look for any electronics models specified"
                }])
            return index, response['message']['content']
        return index, ''

    def reason_prediction(self, max_workers=5):
        self.df['Reason'] = ''
        print("START: Initiating reason behind issues prediction")
        review_counter = 1
        start_time_total = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_reason, index, row): index for index, row in self.df.iterrows()}
            
            for future in concurrent.futures.as_completed(futures):
                index, reason = future.result()
                self.df.at[index, 'Reason'] = reason
                print(f"Review No: {str(index)} completed.")
                review_counter += 1

        end_time_total = time.time()
        total_time = end_time_total - start_time_total

        # Save results to CSV
        self.df.to_csv(self.save_with_reason, index=False)
        print(f"Possible reasons behind issues predicted successfully. Please check ml_predictions_issues_with_reason.csv.")
        print(f"Total time taken to fetch {review_counter - 1} reviews is {total_time:.2f} seconds")
        print("END: Reason behind issues prediction complete")
        print()
       

if __name__ == "__main__":
    obj = ReasonBehindIssuePrediction()
    obj.reason_prediction(max_workers=10)
