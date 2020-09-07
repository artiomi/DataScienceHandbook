import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
ROOT = '/work/workspaces/pycharm/training_data_science/data/'
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df = pd.read_csv(ROOT + 'candy_production.csv', index_col='observation_date', parse_dates=True)

print(df.info())
print(df.isna().any())
# index = pd.to_datetime(df['observation_date'])
# df.index = index
# df.reindex()
# df.drop('observation_date', axis=1, inplace=True)
# df.columns = ['prod_amount']

# resample data
_, ax1 = plt.subplots(figsize=(14, 8))
df.plot(style='-', ax=ax1)
df.resample('AS').mean().plot(style='-o', ax=ax1)
df.asfreq('AS').plot(style=':o', ax=ax1)
ax1.legend(['input', 'resample', 'asfreq'])
ax1.set_title('Compare resample with asfreq')
plt.show()

### Shifting values
_, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 8), sharey='col')
# apply a frequency to the data
freq = df.asfreq('AS', method='pad')
freq.plot(ax=ax1)
freq.shift(10).plot(ax=ax2)
freq.tshift(10).plot(ax=ax3)

# legends and annotations
local_max = pd.to_datetime('1990-01-01')
offset = pd.Timedelta(3650, 'D')

ax1.legend(['input'], loc=2)
ax1.get_xticklabels()[3].set(weight='heavy', color='red')
ax1.axvline(local_max, alpha=0.3, color='red')

ax2.legend(['shift(10AS)'], loc=2)
ax2.get_xticklabels()[3].set(weight='heavy', color='red')
ax2.axvline(local_max + offset, alpha=0.3, color='red')

ax3.legend(['tshift(10AS)'], loc=2)
ax3.get_xticklabels()[3].set(weight='heavy', color='red')
ax3.axvline(local_max + offset, alpha=0.3, color='red')
plt.show()

# rolling statistics
_, ax1 = plt.subplots(figsize=(14, 8))

rolling = df.rolling(12, center=True)
df.plot(ax=ax1, style='-')
rolling.mean().plot(ax=ax1, style='--')
rolling.std().plot(ax=ax1, style=':')
ax1.legend(['input', 'mean', 'std'], loc='upper left')
ax1.lines[0].set_alpha(0.3)
ax1.set_title("Rolling functionality.")
plt.show()

# progress per 10 years
_, ax = plt.subplots(figsize=(14, 8))
yearly = df.resample('AS').sum()
decade = yearly.rolling(10, center=True, win_type='gaussian').sum(std=3)
decade.plot(ax=ax, style='-o')
ax.legend(['progress'], loc='upper left')
ax1.set_title("Decade progress.")
plt.show()

# monthly mean
by_month = df.groupby(df.index.month).mean()
by_month.index = months
plot = by_month.plot()
plt.show()

# heat map of candy production per decade
y_month = df.groupby([df.index.year, df.index.month]).mean()
y_month.index.names = ['year', 'month']
unst = y_month.unstack()
new_index = ((unst.index // 10) * 10).astype(str) + 's'
unst.index = new_index
unst.columns = months
grouped = unst.groupby('year').mean()
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(grouped, annot=True, fmt=".2f", linewidths=.5, ax=ax)
plt.show()
