import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import json
import os
import plotly.io as pio
print(list(pio.templates))


# Path to your JSON file
JSON_PATH = "training_history_gpt.json"

app = dash.Dash(__name__)
app.title = "Live Loss Plot"

app.layout = html.Div([
    html.H2("Training & Validation Loss (Live)"),
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval', interval=2000, n_intervals=0)  # every 2s
])

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval', 'n_intervals')
)
def update_graph(n):
    if not os.path.exists(JSON_PATH):
        return go.Figure()

    with open(JSON_PATH, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return go.Figure()

    train_loss = data.get("train_loss", [])
    val_loss = data.get("val_loss", [])
    x = list(range(1, len(train_loss) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=train_loss, mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=x, y=val_loss, mode='lines+markers', name='Val Loss'))
    fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark')

    return fig

if __name__ == '__main__':
    app.run(debug=False)
