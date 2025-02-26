import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from test_data_model_prediction import ModelPredictions
from test_data_issue_type_prediction import IssueTypePrediction
from sentiment_prediction_bert import SentimentPredictionBERT
from clear_data import ClearAllData
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DashboardApp:
    def __init__(self):
        if sys.platform.startswith("win"):
            self.path = os.path.join(os.getcwd(), "..", "Model_Train_Data_Files")
            self.input_time_file_path = os.path.join(os.getcwd(), "..", "Data_Files")
            self.issues_file = os.path.join(self.path, "bert_predictions_issues_with_reason.csv")
            self.input_time_file = os.path.join(self.input_time_file_path, "input_time.csv")
        elif sys.platform.startswith("darwin"):
            self.path = os.path.join(os.getcwd(), "Model_Train_Data_Files")
            self.input_time_file_path = os.path.join(os.getcwd(), "Data_Files")
            self.issues_file = os.path.join(self.path, "bert_predictions_issues_with_reason.csv")
            self.input_time_file = os.path.join(self.input_time_file_path, "input_time.csv")
        elif sys.platform.startswith("linux"):
            self.path = os.path.join(os.getcwd(), "..", "Model_Train_Data_Files")
            self.input_time_file_path = os.path.join(os.getcwd(), "..", "Data_Files")
            self.issues_file = os.path.join(self.path, "bert_predictions_issues_with_reason.csv")
            self.input_time_file = os.path.join(self.input_time_file_path, "input_time.csv")
        
        self.app = dash.Dash(__name__)
        self.model_prediction = ModelPredictions()
        self.issue_prediction = IssueTypePrediction()
        self.sentiment_prediction = SentimentPredictionBERT()
        self.clear_all_data = ClearAllData()
        self.start_time = None
        self.total_time = 0
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Store(id='graphs-ready', data=False),
            html.Label('Enter Time Range:'),
            dcc.Input(id='time-range', type='text', placeholder='e.g. 25/06/2024 or today'),
            html.Div('Date range can be 12 months, 6 months, 3 months, 30 days, 7 days, 2 days, last day, yesterday, today, this month OR can be custom date: 25/06/2024', style={'marginTop': '10px', 'color': '#0074D9'}),
            html.Button('Generate Predictions', id='button', className='button', style={'backgroundColor': '#0074D9', 'color': 'white', 'marginTop': '10px'}),
            html.Div(id='time-display', children='Time Lapsed: ', style={'marginTop': '10px', 'color': '#0074D9'}),
            dcc.Loading(
                id='loading',
                type='default',
                children=[
                    dcc.Graph(id='graph_model'),
                    dcc.Graph(id='graph_issue'),
                    dcc.Graph(id='graph_issue2'),
                    dcc.Graph(id='graph_issue3')
                ],
                fullscreen=True
            ),
            html.Button('Download CSV', id='download-button', className='button', style={'backgroundColor': '#FF4136', 'color': 'white', 'marginTop': '20px', 'display': 'none'}),
            dcc.Download(id='download')
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('graph_model', 'figure'),
             Output('graph_issue', 'figure'),
             Output('graph_issue2', 'figure'),
             Output('graph_issue3', 'figure'),
             Output('graphs-ready', 'data')],
            [Input('button', 'n_clicks')],
            [State('time-range', 'value')]
        )
        def update_graphs(n_clicks, time_range):
            if n_clicks:
                try:
                    self.start_time = time.time()  # Start timing when button is clicked
                    logging.info(f"Time Range: {time_range}")
                    with open(self.input_time_file, "w") as fo:
                        fo.write(str(time_range).strip())
                    self.clear_all_data.clear_data()
                    logging.info("Generating model prediction...")
                    
                    # Pre-requisites before visual displays
                    self.sentiment_prediction.reviews_extraction_mechanism()  # Step 1: Reviews extraction
                    self.sentiment_prediction.club_reviews()  # Step 2: Club reviews
                    self.sentiment_prediction.clean_and_preprocess_data()  # Step 3: Clean and pre-process data

                    fig_model = self.model_prediction.get_predictions_and_Scores(dash=True)  # Step 4: Model prediction with visuals
                    logging.info("Model prediction generated.")

                    logging.info("Preparing issue prediction visuals...")
                    # Pre-requisites before visual displays
                    self.issue_prediction.pre_requisite_before_visuals()  # Step 5: Issue prediction mechanism

                    fig_sentiment_dist = self.issue_prediction.reviews_sentiment_distribution(dash=True)
                    fig_issue_dist_list = self.issue_prediction.issue_distribution_histogram(dash=True)
                    logging.info("Issue prediction visuals prepared.")

                    # Disabled llama3 since it becomes very slow since it requires 16GB CPU atleast and my PC has only 4 GB spec
                    self.sentiment_prediction.issue_reason_predicted_data() # Step 6: Reason behind issue prediction

                    self.total_time = time.time() - self.start_time  # Calculate total time elapsed
                    return fig_model, fig_sentiment_dist, fig_issue_dist_list[0], fig_issue_dist_list[1], True
                except Exception as e:
                    logging.error(f"Error generating predictions: {e}")
                    return go.Figure(), go.Figure(), go.Figure(), go.Figure(), False
            return go.Figure(), go.Figure(), go.Figure(), go.Figure(), False

        @self.app.callback(
            Output('time-display', 'children'),
            Input('graphs-ready', 'data'),
            prevent_initial_call=True
        )
        def update_time_display(graphs_ready):
            if graphs_ready:
                return f'Total Lapsed Time: {self.format_time(self.total_time)}'
            else:
                if self.start_time:
                    elapsed_time = int(time.time() - self.start_time)
                    return f'Time Lapsed: {self.format_time(elapsed_time)}'
                else:
                    return 'Time Lapsed: 0 seconds'  # Initial state

        @self.app.callback(
            Output('download-button', 'style'),
            Input('graphs-ready', 'data')
        )
        def toggle_download_button(graphs_ready):
            if graphs_ready:
                return {'backgroundColor': '#FF4136', 'color': 'white', 'marginTop': '20px', 'display': 'block'}
            else:
                return {'display': 'none'}

        @self.app.callback(
            Output('download', 'data'),
            Input('download-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def download_csv(n_clicks):
            try:
                if os.path.exists(self.issues_file):
                    return dcc.send_file(self.issues_file)
                else:
                    logging.error("File not found.")
                    return dcc.send_string('File not found', filename='error.txt')
            except Exception as e:
                logging.error(f"Error preparing download: {e}")
                return dcc.send_string('Error preparing download', filename='error.txt')

    def run(self):
        self.app.run_server(debug=True, host="0.0.0.0")

    def format_time(self, seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f'{int(minutes)} min {seconds:.2f} seconds' if minutes > 0 else f'{seconds:.2f} seconds'

if __name__ == '__main__':
    dashboard_app = DashboardApp()
    dashboard_app.run()
