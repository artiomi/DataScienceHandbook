import plotly.express as px

# df = px.data.election()
# geojson = px.data.election_geojson()
#
# fig = px.choropleth(df, geojson=geojson, color="winner",
#                     locations="district", featureidkey="properties.district",
#                     projection="mercator")
# fig.update_geos(fitbounds="locations", visible=False)
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# fig.show()

# countries geometry
# df = px.data.gapminder().query("year==2007")
# fig = px.choropleth(df, locations="iso_alpha",
#                     color="lifeExp",  # lifeExp is a column of gapminder
#                     hover_name="country",  # column to add to hover information
#                     color_continuous_scale=px.colors.sequential.Plasma)
# fig.show()
# print(df.head())
# print(df.columns)

import plotly.express as px

df = px.data.election()
geojson = px.data.election_geojson()

print(df["district"][2])
print(geojson['features'][0])