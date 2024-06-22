# Dataset Libraries
import pandas as pd
import pickle

# Visualization Libraries
import plotly.graph_objects as go
pd.options.display.max_rows = 10
pd.plotting.register_matplotlib_converters()
import warnings
warnings.filterwarnings('ignore')

# Other Libraries
import os
import sys


class ModelPrediction(object):
  def __init__(self):
    if sys.platform.startswith("win"):
        self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        self.df = pd.read_csv(self.path + 'cleaned_test_data_approach3.csv') # Pre-processed test data to pass to saved ML models for prediction
        self.saved_models_path = os.getcwd() + "/../Saved_Models/"
        self.saved_bar_chart_img_path = os.getcwd() + "/../Saved_Images/"
    elif sys.platform.startswith("darwin"):
       self.path = os.getcwd() + "/Model_Train_Data_Files/"
       self.df = pd.read_csv(self.path + 'cleaned_test_data_approach3.csv') # Pre-processed test data to pass to saved ML models for prediction
       self.saved_models_path = os.getcwd() + "/Saved_Models/"
       self.saved_bar_chart_img_path = os.getcwd() + "/Saved_Images/"
    elif sys.platform.startswith("linux"):
        self.path = os.getcwd() + "/../Model_Train_Data_Files/"
        self.df = pd.read_csv(self.path + 'cleaned_test_data_approach3.csv') # Pre-processed test data to pass to saved ML models for predictio 
        self.saved_models_path = os.getcwd() + "/../Saved_Models/"
        self.saved_bar_chart_img_path = os.getcwd() + "/../Saved_Images/"
    
    self.predicted_data = 'final_ml_predictions_approach3.csv'
    self.svc_model_name = "model_svc_test.sav"
    self.lr_model_name = "model_lr_test.sav"
    

    self.df = pd.read_csv(self.path + 'cleaned_test_data_approach3.csv') 

  def loading_saved_model(self, model_type="SVC"):
    # Load the model from the file
    if model_type.upper() in ["SVC", "SUPPORT VECTOR MACHINE"]:
      print("Loading saved SVC model")
      filename = self.saved_models_path + self.svc_model_name
      model_svc = pickle.load(open(filename, 'rb'))
      print(f"{model_type} model loaded successfully")
      return model_svc

    elif model_type.upper() in ["LR", "LOGISTIC REGRESSION"]:
      print("Loading saved LR model")
      filename = self.saved_models_path + self.lr_model_name
      model_lr = pickle.load(open(filename, 'rb'))
      print(f"{model_type} model loaded successfully")
      return model_lr

    else:
      print(f"{model_type} not found in saved models. Please check.")

  def fetch_model_predictions(self, model_type="svc"):
    get_loaded_model = self.loading_saved_model(model_type=model_type)
    X = self.df.Content.values.astype('U')
    predictions = get_loaded_model.predict(X)
    self.df['Prediction'] = predictions
    print(self.df.head())
    print(self.df['Sentiment'].value_counts())
    print(self.df['Prediction'].value_counts())

  def actual_vs_prediction(self, stream_lit=False):
    names_sent = self.df['Sentiment'].value_counts().keys()
    values_sent = self.df['Sentiment'].value_counts().values

    names_pred = self.df['Prediction'].value_counts().keys()
    values_pred = self.df['Prediction'].value_counts().values

    # Create the plot
    fig = go.Figure(data=[
        go.Bar(
            name='True Labels',
            x=names_sent,
            y=values_sent,
            text=values_sent,
            textposition='auto'
        ),
        go.Bar(
            name='Predicted Labels',
            x=names_pred,
            y=values_pred,
            text=values_pred,
            textposition='auto'
        )
    ])

    # Customize layout
    fig.update_layout(
        title='True vs Predicted Labels',
        xaxis_title='Labels',
        yaxis_title='Values',
        barmode='group'
    )
    if stream_lit:
      fig.write_image(self.saved_bar_chart_img_path + "actual_vs_predicted.png", engine='kaleido', scale=1)
    else:
      fig.show()
    
  def main(self, model_type="svc", stream_lit=False):
    print("START: Initiating Model Prediction Pipeline")
    print("######################### STEP 1 #########################")
    print("Load model & Fetch Predictions")
    self.fetch_model_predictions(model_type=model_type)
    print("######################### STEP 2 #########################")
    print("Actual vs Prediction Histogram Graph")
    self.actual_vs_prediction(stream_lit=stream_lit)
    print("######################### STEP 3 #########################")
    print("Save Predictions")
    # Replace values / labels in the 'Prediction' column with actual Sentiments labels
    self.df['Prediction'] = self.df['Prediction'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    self.df.to_csv(self.path + self.predicted_data, index = False)
    print("Predictions saved. Please check final_ml_predictions_approach3.csv")
    print("END: Data prediction complete.")
    print()



if __name__ == "__main__":
  obj = ModelPrediction()
  obj.main(model_type="svc", stream_lit=False)
