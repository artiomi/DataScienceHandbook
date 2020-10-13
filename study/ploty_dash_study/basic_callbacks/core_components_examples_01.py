import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='12', type='number')]),
    html.Br(),
    html.Div(["Input2: ",
              dcc.Input(id='my-input2', value='sample text', type='text')]),
    html.Div(id='my-output'),

])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='my-input', component_property='value'),
     Input(component_id='my-input2', component_property='value')]
)
def update_output_div(input_number, input_value):
    return 'Output value: {}, number {}'.format(input_value,input_number)


if __name__ == '__main__':
    app.run_server(debug=True)