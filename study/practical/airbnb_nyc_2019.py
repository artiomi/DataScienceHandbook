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
def __text_xy_map(bp, orient):
    results = []
    for rect in bp.patches:
        if orient == 'h':
            text = int(rect.get_width())
            xy = (text, rect.get_y() + rect.get_height() / 2)
        elif orient == 'v':
            text = int(rect.get_height())
            xy = ( rect.get_x() + rect.get_width() / 2, text)
        else:
            raise NotImplementedError('no implimentatiton for orient:[{}]'.format(orient))
            
        results.append((text, xy))
    return results

#orient v or h, orientation of barplot
def autolabel(bp, orient='h'):
    text_xy = __text_xy_map(bp,orient)
    
    for text, xy in text_xy:
        bp.annotate('{}'.format(text),
                    xy=xy,
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

#Separating Measures
#quartile
quarantile = location_df[['price', 'minimum_nights']].quantile(q=(.25, .5, .75))
print("quarantile:", quarantile)

#decile
decile = np.linspace(start = 0.1, stop =0.9, num=9)
decile_df = location_df[['price', 'minimum_nights']].quantile(q=decile)
decile_df.index = ['{:2.0f}%'.format(i*100) for i in decile]
print("decile:\n",decile_df)

#percentile
percentile = np.linspace(start = .01, stop =.99, num=99)
percentile_df = location_df[['price', 'minimum_nights']].quantile(q=percentile)
percentile_df.index = ['{:1.0f}%'.format(i*100) for i in percentile]
print("percentile:\n",percentile_df)

#violin plot for minimum nights and price

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2,nrows=2, figsize=(14, 12))

sns.violinplot(y = 'price', data=location_df, ax=ax1, color='cyan')
sns.violinplot(y = 'minimum_nights',  data=location_df, ax=ax2, color='yellow')
sns.violinplot(y = 'price', data=location_df.loc[location_df.price<=750], ax=ax3, color='cyan')
sns.violinplot(y = 'minimum_nights',  data=location_df.loc[location_df.minimum_nights<=32], ax=ax4, color = 'yellow')

ax1.set(title="Price | Violin", ylabel="")
ax2.set(title="Minimum Nights | Violin", ylabel="")
ax3.set(title="Price <= 750 | Violin", ylabel="")
ax4.set(title="Minimum Nights <= 32 | Violin", ylabel="")

ax1.axhline(y = location_df.price.median(), color='black', linestyle= '--', linewidth=2.5)
ax2.axhline(y = location_df.minimum_nights.median(), color='black', linestyle= '--', linewidth=2.5)
ax3.axhline(y = location_df.price.median(), color='black', linestyle= '--', linewidth=2.5)
ax4.axhline(y = location_df.minimum_nights.median(), color='black', linestyle= '--', linewidth=2.5)

#price and minimum nights cumulative plots

f, ax = plt.subplots(ncols=2, figsize=(14, 6))

sns.histplot(location_df, x='price' , bins=70, kde=False, cumulative=True, fill=True,
             stat="density", ax=ax[0],
             alpha=0.9, color="cyan", linewidth= 1.5, edgecolor='black')

sns.histplot(location_df, x='minimum_nights', bins=70, kde=False, cumulative=True, fill=True,  
              stat="density", ax=ax[1],
             alpha= 0.9, color= "yellow",linewidth= 1.5, edgecolor='black')

ax[0].set(title='Price Cumulative', ylabel="Percent (%)",xlabel="Price")

ax[1].set(title='Minimum Nights Cumulative', ylabel="", xlabel="Minimum Nights")

#Dispersion measure

#minimum absolute deviation
mad = location_df[['price', 'minimum_nights']].mad()
print('minimum absolute deviation: ', mad)

#variance
var = location_df[['price', 'minimum_nights']].var()
print("variance: ", var)

#standart deviation
std = location_df[['price', 'minimum_nights']].std()
print('standart deviation: ', std)

mean_room_type = location_df.pivot_table(values='price',index='room_type')

mean_room_type.rename(columns={'price':'average_price'}, inplace=True)
mean_room_type['percent']  = mean_room_type.average_price * 100/mean_room_type.average_price.sum()

#Price behavior in relation to room types

f, ax = plt.subplots(ncols=2, figsize=(14, 6))
sns.histplot(location_df, x='price', hue = 'room_type', bins=70, kde=False, 
             ax=ax[0], legend=False,
             alpha=0.9, color="r",linewidth= 1.5, edgecolor='black')

sns.histplot(location_df[location_df.price<=1000], x='price', hue = 'room_type', bins=70, kde=False, 
             ax=ax[1], legend=True,
             alpha=0.9, color="r",linewidth= 1.5, edgecolor='black')

ax[0].set(xlabel='Price', ylabel='Frequency', title='Price Histogram')
ax[1].set(xlabel='Price', ylabel='', title='Price <= 1000 | Histogram')

f, ax = plt.subplots( figsize=(14, 6))

sns.kdeplot( data=location_df[location_df.price<=1000], ax = ax,
    x="price", hue=location_df.room_type,  multiple="layer", common_norm=False,
    fill=True, bw_adjust=0.45, 
    alpha =.3, linewidth= 1, edgecolor='black' )

ax.set(xlabel='Price', title='Price <= 1000 | Histogram')

#The 10 most expensive neighborhoods to book on airbnb
price_neighbourhood_group = location_df.pivot_table(values='price',index='neighbourhood')
price_neighbourhood_group.sort_values(by='price', ascending=False, inplace=True)
price_neighbourhood_group.reset_index(inplace=True)
top_10_exp_neighbourhood = price_neighbourhood_group[2:12]

#Price behaviour in relation to neighbourhoods
neiborhoods = ["Tribeca", "Sea Gate", "Riverdale", "Prince's Bay", "Battery Park City", "Flatiron District",
               "Randall Manor", "NoHo", "SoHo", "Midtown"]

neigh_prc = location_df.loc[location_df.neighbourhood.isin(neiborhoods),['neighbourhood','price']]
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(neigh_prc, row="neighbourhood",hue="neighbourhood", aspect=15, height=.95,
                  sharex=True, xlim=(-500,4000))

g.map(sns.kdeplot, "price", common_norm=False,  bw_adjust=.25, clip_on=False, fill=True, alpha=.7,
      linewidth=1.5)

g.map(plt.axhline, y=0, lw=2, clip_on=False)


def label(neighbourhood, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="normal", color='black',
             ha="right", va="center", transform=ax.transAxes)
 
 
g.map(label, 'price')

# Set the subplots to overlap
g.fig.subplots_adjust(left=.2, right=.8, bottom = .1, top=.8, hspace=-.65)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(left=True)

#price histogram for 10 most popular neighbourhoods
def modify_axes_locator(*args, **kwargs):
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    
sns.set_theme()
face_grid = sns.FacetGrid(neigh_prc, col="neighbourhood",col_wrap=3, aspect=1.5, height=3,
                          sharex=True, sharey=True, xlim=(0,4000))
face_grid.map(sns.histplot, "price", bins=70,  kde=False,  alpha=.7, linewidth=1.5, edgecolor='black')
face_grid.map(modify_axes_locator)
face_grid.set_titles("{col_name}",fontweight="bold")
face_grid.set_xlabels('')

#The 10 cheapest neighborhoods to book on airbnb
sns.set_theme(palette='Set2')

top_10_cheap_neighbourhood = price_neighbourhood_group.tail(10)
print('The 10 cheapest neighborhoods to book on airbnb:\n', top_10_cheap_neighbourhood)

_, ax = plt.subplots( figsize=(15, 6))
sns.barplot(x= 'neighbourhood', y ='price', data=top_10_cheap_neighbourhood, ax= ax)

p =plt.setp(ax.get_xticklabels(), rotation=30, ha="right",rotation_mode="anchor", fontsize=13)
p =plt.setp(ax.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor", fontsize=13)

ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.set(title='The 10 cheapest neighborhoods to book on airbnb', xlabel='', ylabel='')
autolabel(ax, orient='v')


#Geographic analysis








