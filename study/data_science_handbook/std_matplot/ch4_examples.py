from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import Isomap
from mpl_toolkits.basemap import Basemap
import numpy as np

ROOT = '/work/workspaces/pycharm/training_data_science/data/'

digits = load_digits(n_class=6)
fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])
    
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)

#california cities
cities = pd.read_csv(ROOT+'california_cities.csv')
# Extract the data we're interested in
lat = cities['latd'].values
lon = cities['longd'].values
population = cities['population_total'].values
area = cities['area_total_km2'].values

# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h',lat_0=37.5, lon_0=-119,width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='red')
m.drawcountries(color='gray')
m.drawstates(color='blue')

# 2. scatter city data, with color reflecting population
# and size reflecting area
m.scatter(lon, lat, latlon=True,c=np.log10(population), s=area,cmap='Reds', alpha=0.5)
# 3. create colorbar and legend
plt.colorbar(label=r'$\log_{10}({\rm population})$')
plt.clim(3, 7)
# make legend with dummy points
for a in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,label=str(a) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False,labelspacing=1, loc='lower left')