import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set()

ROOT = '/work/workspaces/pycharm/training_data_science/data/'

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
for col in 'xy':
    plt.hist(data[col],  alpha=0.5)
    

#iris dataset
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species', height=2.5)

#tips histogram
tips = sns.load_dataset('tips')
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))


grid = sns.FacetGrid(tips, row="day", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))

#catplot
with sns.axes_style(style='ticks'):
    g = sns.catplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill")
    
    
#planets analysis
planets = sns.load_dataset('planets')
with sns.axes_style('white'):
    g = sns.catplot("year", data=planets, aspect=2,kind="count", color='steelblue')
    g.set_xticklabels(step=5)
    
    
#marathon data
marathon_data = pd.read_csv(ROOT+'marathon-data.csv')

def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return pd.Timedelta(hours=h, minutes=m, seconds=s)

marathon_data = pd.read_csv(ROOT+'marathon-data.csv',converters={'split':convert_time, 'final':convert_time})
marathon_data['split_sec'] = marathon_data['split'].astype(int) / 1E9
marathon_data['final_sec'] = marathon_data['final'].astype(int) / 1E9

marathon_data['split_frac'] = 1 - 2 * marathon_data['split_sec'] / marathon_data['final_sec']

sns.distplot(marathon_data['split_frac'], kde=False)
plt.axvline(0, color="k", linestyle="--")

#relationship between quantities

g = sns.PairGrid(marathon_data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend()


sns.kdeplot(marathon_data.split_frac[marathon_data.gender=='M'], label='men', shade=True)
sns.kdeplot(marathon_data.split_frac[marathon_data.gender=='W'], label='women', shade=True)
plt.xlabel('split_frac')

marathon_data['age_dec'] = marathon_data.age.map(lambda age: 10 * (age // 10))

men = (marathon_data.gender == 'M')
women = (marathon_data.gender == 'W')
with sns.axes_style(style=None):
    sns.violinplot("age_dec", "split_frac", hue="gender", data=marathon_data,split=True, inner="quartile",palette=["lightblue", "lightpink"])
    
    
g = sns.lmplot('final_sec', 'split_frac', col='gender', data=marathon_data, markers=".", scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.1, color="k", ls=":")








