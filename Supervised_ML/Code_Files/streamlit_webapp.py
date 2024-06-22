import os
import sys
import streamlit as st
import pandas as pd
from sentiment_prediction_ml import SentimentPredictionML


class StreamLitApp(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Saved_Images/"
            self.reason_issue_file_path =  os.getcwd() + "/../Model_Train_Data_Files/"
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Saved_Images/"
            self.reason_issue_file_path =  os.getcwd() + "/Model_Train_Data_Files/"
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Saved_Images/"
            self.reason_issue_file_path = os.getcwd() + "/../Model_Train_Data_Files/"

        self.sentiment_prediction = SentimentPredictionML()

    def main(self):
        st.title("Sentiment Prediction Web App")
        if st.button("Run ML Sentiment Prediction Pipeline"):
            # Pre-requisites before visual displays
            self.sentiment_prediction.reviews_extraction_mechanism()
            self.sentiment_prediction.club_reviews()
            self.sentiment_prediction.clean_and_preprocess_data()

            # Sentiments prediction distribution charts
            st.subheader("Actual vs Predicted Summary")
            self.sentiment_prediction.model_predicted_data()
            
            image = open(self.path + "actual_vs_predicted.png", "rb").read()
            st.image(image, caption='True vs Predicted Labels', use_column_width=True)

            # Reviews and Issues distribution charts
            st.subheader("Trustpilot vs Power reviews Pie Chart")
            self.sentiment_prediction.issue_predicted_data()

            image = open(self.path + "pie_chart.png", "rb").read()
            st.image(image, caption='Reviews Distribution', use_column_width=True)

            st.subheader("Trustpilot Issue Prediction")
            image = open(self.path + "issue_distribution_fig_1.png", "rb").read()
            st.image(image, caption='Trustpilot Issue Distribution', use_column_width=True)
            
            st.subheader("Power Reviews Issue Prediction")
            image = open(self.path + "issue_distribution_fig_2.png", "rb").read()
            st.image(image, caption='Power Reviews Issue Distribution', use_column_width=True)
            
            # Reason behind issue 
            self.sentiment_prediction.issue_reason_predicted_data()
            data = self.reason_issue_file_path + 'ml_predictions_issues_with_reason.csv'
            st.subheader("Issues with reason prediction CSV")
            st.download_button(label="Download", data=data, mime="text/csv")
            

if __name__ == "__main__":
    obj = StreamLitApp()
    obj.main()
