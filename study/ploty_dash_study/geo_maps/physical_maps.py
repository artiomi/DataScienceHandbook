import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure(go.Scattergeo())

# fig = px.line_geo(lat=[0, 15, 20, 35], lon=[5, 10, 25, 30])
fig.update_geos(
    resolution=50, visible=False,
    scope="africa",
    showcountries=True, countrycolor="Black",
    showsubunits=True, subunitcolor="Blue",
    lataxis_showgrid=True, lonaxis_showgrid=True,
    # showcoastlines=True, coastlinecolor="RebeccaPurple",
    # showland=True, landcolor="LightGreen",
    # showocean=True, oceancolor="LightBlue",
    showlakes=True, lakecolor="Blue",
    showrivers=True, rivercolor="Blue",

    # show countries
    #  showcountries=True, countrycolor="RebeccaPurple"
    # fitbounds="locations"

    # map projection
    # , projection_type='gnomonic'
)
fig.update_layout(height=600, margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
