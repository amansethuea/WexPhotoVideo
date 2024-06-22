# Dataset Libraries
import pandas as pd
import numpy as np

# Other Libraries
import os
import sys


class ClubPlatformReviews(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.trust_pilot_data_path = os.getcwd() + "/../Data_Files/trustpilot_all_reviews_api.csv"
            self.power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
            self.trust_pilot_df = pd.read_csv(self.trust_pilot_data_path)
            self.power_reviews_df = pd.read_csv(self.power_reviews_data_path)
            self.fetched_reviews_to_test = os.getcwd() + "/../Model_Train_Data_Files/fetched_test_reviews.csv"
        elif sys.platform.startswith("darwin"):
            self.trust_pilot_data_path = os.getcwd() + "/Data_Files/trustpilot_all_reviews_api.csv"
            self.power_reviews_data_path = os.getcwd() + "/Data_Files/power_reviews_all_api.csv"
            self.fetched_reviews_to_test = os.getcwd() + "/Model_Train_Data_Files/fetched_test_reviews.csv"
            self.trust_pilot_df = pd.read_csv(self.trust_pilot_data_path)
            self.power_reviews_df = pd.read_csv(self.power_reviews_data_path)
    
    def club_reviews(self):
        # Remove unimportant columns from Trust Pilot dataframe
        self.trust_pilot_df.drop(columns=['Title', 'Date of Experience', 'Reviewer Name', 'Location', 'No of reviews given'], inplace=True)
        # Adding new column Source with default value as Trustpilot in Trust Pilot dataframe
        self.trust_pilot_df['Source'] = 'Trustpilot'

        # Remove the same from Power Reviews dataframe and rename columns to match 
        self.power_reviews_df.drop(columns=['Page ID', 'Review Headline', 'Review Location', 'Reviewer Nickname'], inplace=True)
        # Adding new column Source with default value as Trustpilot in Trust Pilot dataframe
        self.power_reviews_df['Source'] = 'PowerReviews'
        # Rename the column names to Content and Rating to keep same in both platforms
        self.power_reviews_df.rename(columns={'Review Rating': 'Rating', 'Comment': 'Content'}, inplace=True)

        # Rename the column Review Date to Created Date to keep same in both platforms
        self.trust_pilot_df.rename(columns={'Review Date': 'Created Date'}, inplace=True) 

        # Club Trust Pilot and Power Reviews Data frames
        combined_df = pd.concat([self.trust_pilot_df, self.power_reviews_df], ignore_index=True)
        
        # Save to csv
        combined_df.to_csv(self.fetched_reviews_to_test, index=False)
        print("Reviews to test successfully clubbed and saved. Please check fetched_test_reviews.csv")

if __name__ == "__main__":
    obj = ClubPlatformReviews()
    obj.club_reviews()