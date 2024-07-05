from trust_pilot_extraction import TrustPilot
from power_reviews_extraction import PowerReviews
from club_platform_reviews import ClubPlatformReviews
from test_data_cleaning_preprocessing import CleanAndPreprocess
from test_data_model_prediction import ModelPredictions
from test_data_issue_type_prediction import IssueTypePrediction
from test_data_reason_behind_issue_prediction import ReasonBehindIssuePrediction
from clear_data import ClearAllData


class SentimentPredictionBERT(object):
    def __init__(self):
        self.clear_all_data = ClearAllData()
        self.trust_pilot = TrustPilot()
        self.power_reviews = PowerReviews()
        self.club_review = ClubPlatformReviews()
        self.clean_and_preprocess = CleanAndPreprocess()
        self.model_prediction = ModelPredictions()
        self.issue_prediction = IssueTypePrediction()
        self.issue_reason = ReasonBehindIssuePrediction()

    def clear_data(self):
        self.clear_all_data.clear_data()
    
    def reviews_extraction_mechanism(self):
        # Trust Pilot
        self.trust_pilot.clear_csv()
        self.trust_pilot.write_csv_data()

        # Power Reviews
        self.power_reviews.get_info()

    def club_reviews(self):
        self.club_review.club_reviews()
    
    def clean_and_preprocess_data(self):
        self.clean_and_preprocess.main(model_type="BERT")
    
    def model_predicted_data(self):
        self.model_prediction.get_predictions_and_Scores()
    
    def issue_predicted_data(self, stream_lit=False):
        self.issue_prediction.main(stream_lit=stream_lit)
    
    def issue_reason_predicted_data(self):
        self.issue_reason.reason_prediction(max_workers=10)
    
    def main(self, stream_lit=False):
        self.clear_data()
        self.reviews_extraction_mechanism()
        self.club_reviews()
        self.clean_and_preprocess_data()
        self.model_predicted_data()
        self.issue_predicted_data(stream_lit=stream_lit)
        self.issue_reason_predicted_data()
    

if __name__ == "__main__":
    obj = SentimentPredictionBERT()
    obj.main(stream_lit=False)
