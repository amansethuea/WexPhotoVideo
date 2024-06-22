from trust_pilot_extraction import TrustPilot
from power_reviews_extraction import PowerReviews
from club_platform_reviews import ClubPlatformReviews
from test_data_cleaning_preprocessing import CleanAndPreprocess
from test_data_model_prediction import ModelPrediction
from test_data_issue_type_prediction import IssueTypePrediction
from test_data_reason_behind_issue_prediction import ReasonBehindIssuePrediction


class SentimentPredictionML(object):
    def __init__(self):
        self.trust_pilot = TrustPilot()
        self.power_reviews = PowerReviews()
        self.club_review = ClubPlatformReviews()
        self.clean_and_preprocess = CleanAndPreprocess()
        self.model_prediction = ModelPrediction()
        self.issue_prediction = IssueTypePrediction()
        self.issue_reason = ReasonBehindIssuePrediction()

    def reviews_extraction_mechanism(self):
        # Trust Pilot
        self.trust_pilot.clear_csv()
        self.trust_pilot.write_csv_data()

        # Power Reviews
        self.power_reviews.get_info()

    def club_reviews(self):
        self.club_review.club_reviews()
    
    def clean_and_preprocess_data(self):
        self.clean_and_preprocess.main(model_type="ML")
    
    def model_predicted_data(self):
        self.model_prediction.main(model_type="svc")
    
    def issue_predicted_data(self):
        self.issue_prediction.main()
    
    def issue_reasdon_predicted_data(self):
        self.issue_reason.reason_prediction(max_workers=10)
    
    def main(self):
        self.reviews_extraction_mechanism()
        self.club_reviews()
        self.clean_and_preprocess_data()
        self.model_predicted_data()
        self.issue_predicted_data()
        self.issue_reasdon_predicted_data()
    

if __name__ == "__main__":
    obj = SentimentPredictionML()
    obj.main()