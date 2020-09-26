import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

ROOT = '/work/workspaces/pycharm/training_data_science/data/'
df = pd.read_csv(ROOT+'AB_NYC_2019.csv')

location_df = df[['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'price','room_type','minimum_nights']]

print('neighbourhood_group \n',location_df.neighbourhood_group.value_counts())
print('room_type \n',location_df.room_type.value_counts())
print('numeric columns \n',location_df.describe())

print(location_df.neighbourhood.unique())

# Qualitative Variables Frequency Distribution Tables
neighb_grp_freq_distr = pd.DataFrame({'frequency':location_df.neighbourhood_group.value_counts(),
                                      'percent':(location_df.neighbourhood_group.value_counts() *100/ location_df.neighbourhood_group.value_counts().sum()).round(decimals=2)})

neighbourhood_freq_distr = pd.DataFrame({'frequency':location_df.neighbourhood.value_counts(),
                                         'percent':(location_df.neighbourhood.value_counts() *100/ location_df.neighbourhood.value_counts().sum())})

room_type_freq_distr = pd.DataFrame({'frequency':location_df.room_type.value_counts(),
                                     'percent':(location_df.room_type.value_counts() *100/ location_df.room_type.value_counts().sum()).round(decimals=2)})

n = 5
#getting first n elemnents with highets frequency
neighb_grp_freq_distr.sort_values('percent', ascending=False).head(n)
room_type_freq_distr.sort_values('percent', ascending=False).head(n)
neighbourhood_freq_distr.sort_values('percent', ascending=False).head(n)

# Frequency Distribution Tables for quantitative variables

#Sturges Rule
k = np.around(1 + ((10/3) * np.log10(location_df.size))).astype(int)

nights_labels =[
    '      1  |—|  66.737', 
    ' 66.737   —|  132.474', 
    ' 132.474  —|  198.211', 
    ' 198.211  —|  263.947', 
    ' 263.947  —|  329.684', 
    ' 329.684  —|  395.421', 
    ' 395.421  —|  461.158', 
    ' 461.158  —|  526.895', 
    ' 526.895  —|  592.632', 
    ' 592.632  —|  658.368', 
    ' 658.368  —|  724.105', 
    ' 724.105  —|  789.842', 
    ' 789.842  —|  855.579', 
    ' 855.579  —|  921.316', 
    ' 921.316  —|  987.053', 
    ' 987.053  —| 1052.789', 
    ' 1052.789 —| 1118.526',
    ' 1118.526 —| 1184.263', 
    ' 1184.263 —| 1250' ]

splitted_min_nights_df = pd.cut(location_df.minimum_nights, bins=k,labels=nights_labels, include_lowest=True)
min_nights_freq_distr = pd.DataFrame({'frequency':splitted_min_nights_df.value_counts().sort_index(),
                                      'percent':(splitted_min_nights_df.value_counts() * 100/splitted_min_nights_df.value_counts().sum()).sort_index()})


price_labels =[
    '      1  |—|  526.316', 
    ' 526.316  —|  1052.632', 
    ' 1052.632 —| 1578.947', 
    ' 1578.947 —| 2105.263', 
    ' 2105.263 —| 2631.579', 
    ' 2631.579 —| 3157.895', 
    ' 3157.895 —| 3684.211', 
    ' 3684.211 —| 4210.526', 
    ' 4210.526 —| 4736.842', 
    ' 4736.842 —| 5263.158', 
    ' 5263.158 —| 5789.474', 
    ' 5789.474 —| 6315.789', 
    ' 6315.789 —| 6842.105', 
    ' 6842.105 —| 7368.421', 
    ' 7368.421 —| 7894.737', 
    ' 7894.737 —| 8421.053', 
    ' 8421.053 —| 8947.368',
    ' 8947.368 —| 9473.684', 
    ' 9473.684 —| 10000.0' ]

splitted_price_df = pd.cut(location_df.price, bins=k,labels=price_labels, include_lowest=True)
min_price_freq_distr = pd.DataFrame({'frequency':splitted_price_df.value_counts().sort_index(),
                                      'percent':(splitted_price_df.value_counts() * 100/splitted_price_df.value_counts().sum()).sort_index()})


#Prices frequency histogram
f, ax = plt.subplots(ncols=2, figsize=(14, 6))

sns.distplot(location_df.price , bins=70, kde=False,  ax=ax[0],
             hist_kws={"alpha": 0.9, "color": "r","linewidth": 1.5, 'edgecolor':'black'})
ax[0].set(title='Total prices frequency', ylabel="Frequency",xlabel="Price")

sns.distplot(location_df.loc[location_df.price<500, 'price'] , bins=70, kde=False,  ax=ax[1],
             hist_kws={"alpha": 0.9, "color": "r","linewidth": 1.5, 'edgecolor':'black'})

ax[1].set(title='(Prices<500) frequency', ylabel="Frequency", xlabel="Price")

#Nights frequency histogram
f, ax = plt.subplots(ncols=2, figsize=(14, 6))

sns.distplot(location_df.minimum_nights , bins=70, kde=False,  ax=ax[0],
             hist_kws={"alpha": 0.9, "color": "g","linewidth": 1.5, 'edgecolor':'black'})
ax[0].set(title='Total minimum nights', ylabel="Frequency",xlabel="Min. nights")

sns.distplot(location_df.loc[location_df.minimum_nights<35, 'minimum_nights'] , bins=70, kde=False,  ax=ax[1],
             hist_kws={"alpha": 0.9, "color": "g","linewidth": 1.5, 'edgecolor':'black'})

ax[1].set(title='(Min. nights<35) frequency', ylabel="Frequency", xlabel="Min. nights")




