import streamlit as st
from sentiment_prediction_ml import SentimentPredictionML


class StreamLitApp(object):
    def __init__(self):
        self.sentiment_prediction = SentimentPredictionML()
    

    def main(self):
        st.title("Sentiment Prediction Web App")
        if st.button("Run ML Sentiment Prediction Pipeline"):
            self.sentiment_prediction.main()

            # Display Results
            st.subheader("Results Summary")
            self.sentiment_prediction.main()


if __name__ == "__main__":
    obj = StreamLitApp()
    obj.main()