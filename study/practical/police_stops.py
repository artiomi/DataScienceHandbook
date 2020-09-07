import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

sns.set()
ROOT = '/work/workspaces/pycharm/training_data_science/data/'
df = pd.read_csv(ROOT + 'police.csv')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

df.rename(
    columns={'driver_age': 'age', 'driver_age_raw': 'year_of_birth', 'driver_race': 'race', 'driver_gender': 'gender'},
    inplace=True)

print(df.info())

df.drop('county_name', axis=1, inplace=True)

df.groupby('gender')['gender'].count()
df['gender'].fillna(method='pad', inplace=True)

# search type is related to
df[df['search_type'].notna() & df['search_conducted']].count()
df.loc[df['search_type'].isna(), ['search_type']] = 'NO SEARCH'

# drop NaNs because there is same number in almost all columns
df.dropna(inplace=True)
df['age'] = df['age'].astype(np.int)
df['year_of_birth'] = df['year_of_birth'].astype(np.int)
df['is_arrested'] = df['is_arrested'].astype(np.bool)

df.stop_duration.replace(['1','2'],'30+ Min', inplace=True )

mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
df['stop_minutes'] = df.stop_duration.map(mapping)

# add multiindex from dates
# mi = pd.MultiIndex.from_frame(df[['stop_date','stop_time']])
df.index = pd.to_datetime(df['stop_date'] + ' ' + df['stop_time'])
df.index.name='stop_date_time'
df.drop(['stop_date', 'stop_time'], axis=1, inplace=True)

#group by month and gender
month_gen = df.groupby([df.index.month,'gender'])['gender'].count().unstack()
month_gen.index = months
month_gen.plot()
plt.show()

#get hourly stop by gender
hour_gen = df.groupby([df.index.hour,'gender'])['gender'].count().unstack()
hour_gen.plot()
plt.show()

#bar plot by violation and age
age_periods= pd.cut(df['age'],5, precision=1)
violation_pt = df.pivot_table(df[['stop_outcome']],index='violation',columns=age_periods, aggfunc='count')
violation_pt.plot(kind='bar')
plt.show()

#return violation kind
df.groupby(['violation', 'stop_outcome'])['violation'].count().unstack(level=1).plot(kind='bar')


#group violations by age group
us = df.groupby([age_periods,'violation'])['violation'].count().unstack(level=0)
per_age_proc = us.transform((lambda x: (x/x.sum()) * 100), axis=0).round(1)
per_age_proc.T.plot.bar()
plt.show()

#per violation
per_viol = us.transform((lambda x: (x/x.sum()) * 100), axis=1).round(1)
per_viol.T.plot.bar()
plt.show()

#relationship between violation and stop outcome
viol_outcome_mins = df.pivot_table(values='stop_minutes', columns='stop_outcome', index='violation').round(1)
pd.concat([viol_outcome_mins.idxmin(), viol_outcome_mins.min()], axis=1,keys=['violation', 'time'])

####currently not used cleaning functions

# fix driver raw age, replace all Nan, missing and not relevant with mean date
# df['year_of_birth'].fillna(0, inplace=True)
# df['year_of_birth'] = df['year_of_birth'].astype(np.int)
# df.loc[df['year_of_birth'] > 2020, 'year_of_birth'] = 0
# df.loc[df['year_of_birth'] < 1920, 'year_of_birth'] = 0
# print(df.loc[df['year_of_birth'] > 0, 'year_of_birth'].describe())
#
# # set missing year of birth to mean
# df.loc[df['year_of_birth'] == 0, 'year_of_birth'] = round(df.loc[df['year_of_birth'] > 0, 'year_of_birth'].mean())

# fixing age
# df['age'].fillna(0, inplace=True)
# df['age'] = df['age'].astype(np.int)
# df.loc[df['age'] > 100, 'age'] = 0
# df.loc[df['age'] < 0, 'age'] = 0
# df.loc[df['age'] == 0, 'age'] = round(df.loc[df['age'] > 0, 'age'].mean())
#
# # process race
# df.groupby('race')['race'].count()
# df['race'].describe()
# df['race'].fillna(method='pad', inplace=True)
#
# # because volation an violation raw have equal nr of null check for intersections
# df.loc[(df['violation'].notna() & df['violation_raw'].notna()), ['violation_raw', 'violation']].count()
#
# # violation raw
# df.groupby('violation_raw')['violation_raw'].count()
# df['violation_raw'].describe()
# df['violation_raw'].fillna(method='pad', inplace=True)
#
# # violation
# df.groupby('violation')['violation'].count()
# df['violation'].describe()
# df['violation'].fillna(method='pad', inplace=True)








