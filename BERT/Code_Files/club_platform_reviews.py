# Dataset Libraries
import pandas as pd

# Other Libraries
import os
import sys


class ClubPlatformReviews(object):
    def club_reviews(self):
        if sys.platform.startswith("win"):
            trust_pilot_data_path = os.getcwd() + "/../Data_Files/trustpilot_all_reviews_api.csv"
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("darwin"):
            trust_pilot_data_path = os.getcwd() + "/Data_Files/trustpilot_all_reviews_api.csv"
            power_reviews_data_path = os.getcwd() + "/Data_Files/power_reviews_all_api.csv"
        elif sys.platform.startswith("linux"):
            trust_pilot_data_path = os.getcwd() + "/../Data_Files/trustpilot_all_reviews_api.csv"
            power_reviews_data_path = os.getcwd() + "/../Data_Files/power_reviews_all_api.csv"
        
        trust_pilot_df = pd.read_csv(trust_pilot_data_path)
        power_reviews_df = pd.read_csv(power_reviews_data_path)

        # Remove unimportant columns from Trust Pilot dataframe
        trust_pilot_df.drop(columns=['Title', 'Date of Experience', 'Reviewer Name', 'Location', 'No of reviews given'], inplace=True)
        # Adding new column Source with default value as Trustpilot in Trust Pilot dataframe
        trust_pilot_df['Source'] = 'Trustpilot'

        # Remove the same from Power Reviews dataframe and rename columns to match 
        power_reviews_df.drop(columns=['Page ID', 'Review Headline', 'Review Location', 'Reviewer Nickname'], inplace=True)
        # Adding new column Source with default value as Trustpilot in Trust Pilot dataframe
        power_reviews_df['Source'] = 'PowerReviews'
        # Rename the column names to Content and Rating to keep same in both platforms
        power_reviews_df.rename(columns={'Review Rating': 'Rating', 'Comment': 'Content'}, inplace=True)

        # Rename the column Review Date to Created Date to keep same in both platforms
        trust_pilot_df.rename(columns={'Review Date': 'Created Date'}, inplace=True) 

        # Club Trust Pilot and Power Reviews Data frames
        combined_df = pd.concat([trust_pilot_df, power_reviews_df], ignore_index=True)
        
        # Save to csv
        if sys.platform.startswith("win"):
            fetched_reviews_to_test = os.getcwd() + "/../Model_Train_Data_Files/fetched_test_reviews.csv"
        elif sys.platform.startswith("darwin"):
            fetched_reviews_to_test = os.getcwd() + "/Model_Train_Data_Files/fetched_test_reviews.csv"
        elif sys.platform.startswith("linux"):
            fetched_reviews_to_test = os.getcwd() + "/../Model_Train_Data_Files/fetched_test_reviews.csv"

        combined_df.to_csv(fetched_reviews_to_test, index=False)
        print("Reviews to test successfully clubbed and saved. Please check fetched_test_reviews.csv")

if __name__ == "__main__":
    obj = ClubPlatformReviews()
    obj.club_reviews()
