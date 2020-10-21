import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

ROOT = '/work/workspaces/pycharm/training_data_science/data/'
df = pd.read_csv(ROOT + 'AB_NYC_2019.csv')
# df = px.data.election()
# geojson = px.data.election_geojson()
#
# fig = px.choropleth(df, geojson=geojson, color="winner",
#                     locations="district", featureidkey="properties.district",
#                     projection="mercator")
# fig.update_geos(fitbounds="locations", visible=False)
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# fig.show()

fig = go.Figure(data=go.Scattergeo(
    lon=df['longitude'],
    lat=['latitude'],
    mode='markers',
locations=df['neighbourhood_group']

))

fig.update_layout(
    title='Most trafficked US airports<br>(Hover for airport names)',
    geo_scope='usa',
)
fig.show()
#print(df[['neighbourhood','neighbourhood_group']])