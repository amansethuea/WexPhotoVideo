import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from test_data_model_prediction import ModelPrediction
from test_data_issue_type_prediction import IssueTypePrediction
import webbrowser

app = dash.Dash(__name__)

# Initialize the ModelPrediction and IssueTypePrediction classes
model_prediction = ModelPrediction()
issue_prediction = IssueTypePrediction()

app.layout = html.Div([
    html.Button('Generate Predictions', id='button', className='button', style={'backgroundColor': '#0074D9', 'color': 'white'}),
    dcc.Graph(id='graph_model'),
    dcc.Graph(id='graph_issue'),
    dcc.Graph(id='graph_issue2'),
    dcc.Graph(id='graph_issue3')
])

@app.callback(
    [Output('graph_model', 'figure'),
     Output('graph_issue', 'figure'),
     Output('graph_issue2', 'figure'),
     Output('graph_issue3', 'figure')],
    Input('button', 'n_clicks')
)
def update_graphs(n_clicks):
    if n_clicks:
        # Generate figures for both model and issue predictions
        fig_model = model_prediction.main(dash=True)
        issue_prediction.pre_requisite_before_visuals()
        fig_sentiment_dist = issue_prediction.reviews_sentiment_distribution(dash=True)
        fig_issue_dist_list = issue_prediction.issue_distribution_histogram(dash=True)
        return fig_model, fig_sentiment_dist, fig_issue_dist_list[0], fig_issue_dist_list[1]
    return go.Figure(), go.Figure(), go.Figure(), go.Figure()

if __name__ == '__main__':
    # Automatically open the default web browser
    # webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=True)
