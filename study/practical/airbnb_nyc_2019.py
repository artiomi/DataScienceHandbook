import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
sns.set()
sns.set_theme(palette='Set2')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

ROOT = '/work/workspaces/pycharm/training_data_science/data/'
df = pd.read_csv(ROOT+'AB_NYC_2019.csv')

location_df = df[['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'price','room_type','minimum_nights']]

print('neighbourhood_group \n',location_df.neighbourhood_group.value_counts())
print('room_type \n',location_df.room_type.value_counts())
print('numeric columns \n',location_df.describe())


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

sns.histplot(location_df, x='price' , bins=70, kde=False,  ax=ax[0],
             alpha=0.9, color="r",linewidth= 1.5, edgecolor='black')
ax[0].set(title='Total prices frequency', ylabel="Frequency",xlabel="Price")

sns.histplot(location_df.loc[location_df.price<500], x='price'  , bins=70, kde=False,  ax=ax[1],
             alpha= 0.9, color= "r",linewidth= 1.5, edgecolor='black')

ax[1].set(title='(Prices<500) frequency', ylabel="Frequency", xlabel="Price")

#Nights frequency histogram
f, ax = plt.subplots(ncols=2, figsize=(14, 6))

sns.histplot(location_df, x='minimum_nights' , bins=70, kde=False,  ax=ax[0],
             alpha=0.9, color="g",linewidth= 1.5, edgecolor='black')
ax[0].set(title='Total minimum nights', ylabel="Frequency",xlabel="Min. nights")

sns.histplot(location_df.loc[location_df.minimum_nights<35],x='minimum_nights' , bins=70, kde=False,  ax=ax[1],
             alpha=0.9, color="g",linewidth= 1.5, edgecolor='black')

ax[1].set(title='(Min. nights<35) frequency', ylabel="Frequency", xlabel="Min. nights")

#function for add labels to barplot
def autolabel(bp):
    for rect in bp.patches:
        width = int(rect.get_width())       
        bp.annotate('{}'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='center', bbox=(dict(facecolor='green', alpha=0.8)),
                    color='white', family='monospace')
#Room type distribution
f, ax = plt.subplots( figsize=(9, 5),subplot_kw=dict(facecolor='#EEE8AA'), facecolor='#EEE8AA')
bp = sns.barplot(x="frequency",y = room_type_freq_distr.index, data=room_type_freq_distr, ax = ax,
                 capstyle ='round')
ax.set_title("Room type distribution")
autolabel(bp)

#the most frequent neighborhood
top_10_neighborhood = neighbourhood_freq_distr.sort_values('frequency',ascending=False).head(10).sort_index()
f, ax = plt.subplots( figsize=(15, 6))
sns.barplot(x= top_10_neighborhood.index, y = top_10_neighborhood.frequency, data=top_10_neighborhood, ax= ax)
sns.lineplot(x= top_10_neighborhood.index, y = top_10_neighborhood.frequency, data=top_10_neighborhood,
             ax= ax, linestyle='--', color='black',alpha = .5, marker="o", markersize=10.,
             markerfacecolor='green')
p =plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor", fontsize=13)
p =plt.setp(ax.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor", fontsize=13)

ax.legend(ax.patches, [l.get_text() for l in ax.get_xticklabels()],title = 'neghbourhoods',
          loc='upper right',bbox_to_anchor=(1.24,1), fontsize=13)
ax.set_title('The 10 most frequent neighbourhood')

#neighbourhood group distribution
f, ax = plt.subplots( figsize=(9, 5),subplot_kw=dict(facecolor='#DCFCE6'), facecolor='#DCFCE6')
bp = sns.barplot(x="frequency",y = neighb_grp_freq_distr.index, data=neighb_grp_freq_distr, ax = ax)
ax.set_title("Location Distribution")
autolabel(bp)

## Central Trend Measures
#mean
location_df[['price', 'minimum_nights']].mean()

#median
location_df[['price', 'minimum_nights']].median()

#mode
location_df[['price', 'minimum_nights']].mode()
def format_func(value, tick_number):
    if tick_number%2==0:
        result = int(value) if value<=0 or value>=1 else round(value, 3)
        return result
    else:
        return ''
    

#median plots
f, ax = plt.subplots(ncols=2, figsize=(14, 6))

sns.kdeplot(data=location_df.loc[location_df.price<=750], x='price', fill=True,
           color = 'red',ax = ax[0], alpha=.2, edgecolor='black')
sns.kdeplot(data=location_df.loc[location_df.minimum_nights<=32], x='minimum_nights',
            color = 'blue', fill=True, ax = ax[1], alpha=.2, edgecolor='black')

#mark mean, median and mode
ax[0].axvline(x = location_df.price.mean(), color='black', linestyle= '-.', linewidth=1.5)
ax[0].axvline(x = location_df.price.median(), color='red', linestyle= '--', linewidth=1.5)
ax[0].axvline(x = location_df.price.mode()[0], color='green', linestyle= ':', linewidth=2.5)

ax[1].axvline(x = location_df.minimum_nights.mean(), color='black', linestyle= '-.', linewidth=1.5)
ax[1].axvline(x = location_df.minimum_nights.median(), color='red', linestyle= '--', linewidth=1.5)
ax[1].axvline(x = location_df.minimum_nights.mode()[0], color='green', linestyle= ':', linewidth=2.5)

#add texts for mean, median and mode
ax[0].text(s="Mean = 152.72", x = 270, y = .0065, color='black', fontsize =13, fontweight='semibold')
ax[0].text(s="Median = 106", x = 270, y = .0055, color='red', fontsize =13, fontweight='semibold')
ax[0].text(s="Mode = 100", x = 270, y = .0045, color='green', fontsize =13, fontweight='semibold')

ax[1].text(s="Mean = 7.02", x = 12, y = .185, color='black', fontsize =13, fontweight='semibold')
ax[1].text(s="Median = 3", x = 12, y = .165, color='red', fontsize =13, fontweight='semibold')
ax[1].text(s="Mode = 1", x = 12, y = .135, color='green', fontsize =13, fontweight='semibold')


#ax[0].xaxis.set_major_locator(plt.IndexLocator(offset=ax[0].xaxis.get_data_interval()[0] * -1, base=200))
#ax[0].yaxis.set_major_locator(plt.IndexLocator(offset=0, base=0.002))

ax[0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(format_func))

#ax[1].xaxis.set_major_locator(plt.IndexLocator(offset=ax[1].xaxis.get_data_interval()[0] * -1, base=10))
#ax[1].yaxis.set_major_locator(plt.IndexLocator(offset=0, base=0.05))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(format_func))

ax[0].set_title('Price <= 750 | Density')
ax[0].set_xlim(0,750)

ax[1].set_title('Minimum nights <= 32 | Density')
ax[1].set_xlim(0,32)
