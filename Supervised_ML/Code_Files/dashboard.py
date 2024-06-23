import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from test_data_model_prediction import ModelPrediction
import webbrowser

app = dash.Dash(__name__)

# Initialize the ModelPrediction class
model_prediction = ModelPrediction()

app.layout = html.Div([
    html.Button('Supervised ML Predictions', id='button'),
    dcc.Graph(id='graph')
])

@app.callback(
    Output('graph', 'figure'),
    Input('button', 'n_clicks')
)
def update_graph(n_clicks):
    if n_clicks:
        # Generate the figure using the actual_vs_prediction function
        fig = model_prediction.main(dash=True)
        return fig
    return go.Figure()

if __name__ == '__main__':
    # Automatically open the default web browser
    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=True)
