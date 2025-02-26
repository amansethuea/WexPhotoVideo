# Dataset Libraries
import pandas as pd
import numpy as np

# Preprocessing Libraries
import spacy
from spacy.matcher import PhraseMatcher

# Visualization Libraries
import plotly.graph_objects as go
pd.options.display.max_rows = 10
pd.plotting.register_matplotlib_converters()
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Other Libraries
import os
import random
import sys


class IssueTypePrediction(object):
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'final_ml_predictions_approach3.csv') # Data with predictions
            self.issue_predicted_data = self.path + 'ml_predictions_with_issue_types.csv' # Data with issues prediction
            self.saved_pie_chart_img_path = os.getcwd() + "/../Saved_Images/"
        elif sys.platform.startswith("darwin"):
            self.path = os.getcwd() + "/Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'final_ml_predictions_approach3.csv') # Data with predictions
            self.issue_predicted_data = self.path + 'ml_predictions_with_issue_types.csv' # Data with issues prediction
            self.saved_pie_chart_img_path = os.getcwd() + "/Saved_Images/"
        elif sys.platform.startswith("linux"):
            self.path = os.getcwd() + "/../Model_Train_Data_Files/"
            self.df = pd.read_csv(self.path + 'final_ml_predictions_approach3.csv') # Data with predictions
            self.issue_predicted_data = self.path + 'ml_predictions_with_issue_types.csv' # Data with issues prediction
            self.saved_pie_chart_img_path = os.getcwd() + "/../Saved_Images/"

        self.nlp = spacy.load("en_core_web_lg")
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self.issue_phrases = {
            "Electronics": ["laptop", "camera", "lens", "mobile phone", "smartphone",
                            "tripod", "tablet", "headphone", "charger", "battery"],
            "Product": ["product issue", "defective product", "faulty item",
                        "damaged", "damage", "torn", "broken", "break", "gouged"],
            "Packaging": ["packaging", "package", "pack"],
            "Time": ["late", "delayed", "time issue", "too long", "long to arrive"],
            "Delivery": ["delivery issue", "shipment problem", "shipping issue", "arrive",
                         "arrival", "late delivery", "delivery late", "late", "time", "times"],
            "Customer service": ["customer service", "support issue", "service complaint",
                                 "rude", "aggressive", "tone", "behaviour", "nonsense",
                                 "not understand", "staff", "insult", "salesperson"],
            "Payment": ["payment", "pay", "money"],
            "General": ["issue", "problem", "complaint"]
        }
        self.phrase_matcher()

    def phrase_matcher(self):
        for label, phrases in self.issue_phrases.items():
            patterns = [self.nlp.make_doc(text) for text in phrases]
            self.matcher.add(label, patterns)

    def classify_issues(self, text):
        if not isinstance(text, str):
            text = str(text)
        doc = self.nlp(text)
        matches = self.matcher(doc)
        issues = set()

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            if label != "General":
                issues.add(label)
        
        # If no other issues are found, check for 'General'
        if not issues:
            for phrase in self.issue_phrases["General"]:
                if phrase in text:
                    issues.add("General")
                    break
                else:
                    issues.add("No Issue")
        
        return issues

    def reviews_sentiment_distribution(self, stream_lit=False, dash=False):
      df = pd.read_csv(self.issue_predicted_data)

      filtered_df = df[df['Source'].isin(['Trustpilot', 'PowerReviews'])]
      fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Trustpilot', 'PowerReviews'])

      # Add pie charts to the subplot
      for i, source in enumerate(['Trustpilot', 'PowerReviews']):
          source_data = filtered_df[filtered_df['Source'] == source]['Prediction'].value_counts().reset_index()
          source_data.columns = ['Prediction', 'Count']
        
          fig.add_trace(
              go.Pie(
                  labels=source_data['Prediction'],
                  values=source_data['Count'],
                  hole=0.4,
                  textinfo='percent+label',
                  hoverinfo='label+value'
              ),
              row=1, col=i+1
          )
      
      if stream_lit:
          fig.write_image(self.saved_pie_chart_img_path + "pie_chart.png", engine='kaleido', scale=1)
      elif dash:
          fig.update_layout(title_text='Distribution of Reviews for Trustpilot and PowerReviews')
          return fig
      else:     
        fig.update_layout(title_text='Distribution of Reviews for Trustpilot and PowerReviews')
        fig.show()
      

    def get_issue_distribution_per_platform(self):
        df = pd.read_csv(self.issue_predicted_data)
        # Dropping empty issues records first
        df = df.dropna(subset=['Issues'])
    
        source_dict = {}
        for _, row in df.iterrows():
            source = row['Source']
            
            if source == 'PowerReviews' or source == 'Trustpilot':
                issues_set = eval(row['Issues'])
                if source in source_dict:
                    source_dict[source].extend(issues_set)
                else:
                    source_dict[source] = list(issues_set)
        
        return source_dict

    def get_issue_distribution_per_category(self):
        source_dict_modified = {}
        source_dict = self.get_issue_distribution_per_platform()
        for source, issues_list in source_dict.items():
            issue_count = {}
            for issue in issues_list:
                if issue in issue_count:
                    issue_count[issue] += 1
                else:
                    issue_count[issue] = 1
            source_dict_modified[source] = issue_count
        
        return source_dict_modified
    
    def issue_distribution_histogram(self, stream_lit=False, dash=False):
    
        figures, dash_fig_list = [], []
        source_dict_modified = self.get_issue_distribution_per_category()
        
        # Define a list of colors to use for the bars
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

        for source, issue_counts in source_dict_modified.items():
            issues = list(issue_counts.keys())
            counts = list(issue_counts.values())
            
            if len(issues) > len(colors):
                additional_colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(len(issues) - len(colors))]
                colors.extend(additional_colors)

            bar_colors = colors[:len(issues)]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=issues, y=counts, marker_color=bar_colors))

            fig.update_layout(
                title=f"Issues Count for {source}",
                xaxis_title="Issues",
                yaxis_title="Count",
                showlegend=False
            )

            figures.append(fig)
        
        fig_counter = 1
        for fig in figures:
            if stream_lit:
                fig.write_image(self.saved_pie_chart_img_path + f"issue_distribution_fig_{str(fig_counter)}.png", engine='kaleido', scale=1)
                fig_counter += 1
            elif dash:
                dash_fig_list.append(fig)
            else:
                fig.show()
        if dash:
            return dash_fig_list

    def pre_requisite_before_visuals(self):
        # Fresh df read
        self.df = pd.read_csv(self.path + 'final_ml_predictions_approach3.csv')
        print("START: Initiating predicting Issue Type")
        filtered_df = self.df[self.df['Prediction'].isin(['Positive', 'Negative', 'Neutral'])]
        filtered_df['Issues'] = filtered_df['Content'].apply(self.classify_issues)
        filtered_df.to_csv(self.issue_predicted_data, index=False)
        print("Issue Types predicted successfully. Please check ml_predictions_with_issue_types.csv")
    
    def main(self, stream_lit=False, dash=False):
        self.pre_requisite_before_visuals()
        print("Reviews Distribution Pie Chart.")
        self.reviews_sentiment_distribution(stream_lit=stream_lit, dash=dash)
        print("Issue Distribution Histogram.")
        self.issue_distribution_histogram(stream_lit=stream_lit, dash=dash)
        print("END: Issues predicted complete.")
        print()

if __name__ == "__main__":
    obj = IssueTypePrediction()
    obj.main(stream_lit=False, dash=False)
