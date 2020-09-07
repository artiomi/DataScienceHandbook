import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

ROOT = '/work/workspaces/pycharm/training_data_science/data/'

crash_data = pd.read_csv(ROOT + "car_crashes.csv")
# 1. verify if dataframe contains nones
print(crash_data.info())
print(crash_data.isna().any())


# no None entry found

# 2. Function to find categorical and non-categorical features
def cat_finder(df):
    cat = []
    cont = []
    cols = list(df.columns)
    for col in cols:
        t = df[col].dtype  # returns the type of the feature
        print(f'Col:{col} type: {t}', col, t)
        if t == 'O':
            cat.append(col)
        else:
            cont.append(col)

    return cat, cont


# Using the function above
cat, cont = cat_finder(crash_data)
print("\nCategorical Features:", cat)
print("\nContinuous Features:\n", cont)

# 3.show correlation matrix
# Visualize using heatmap
corr = crash_data[cont].corr()
plt.figure(figsize=(7, 4))
plt.title("The Correlations of the Features")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, mask=mask, cmap="inferno")
plt.show()

# correlation between alcohol and no_previous (0,78)
sns.lmplot(x='alcohol', y='no_previous', fit_reg=False, scatter_kws={"color": "blue", "alpha": 0.5, "s": 100},
           data=crash_data)
plt.xlabel('Alcohol consumption', size=8)
plt.ylabel('No previous accidents', size=8)
plt.title("Relation between Alcohol and not previous\n", fontsize=10)
plt.show()

# correlation between alcohol, not_distracted and no_previous (0,73)
sns.lmplot(x='speeding', y='no_previous', fit_reg=False, scatter_kws={"color": "blue", "alpha": 0.5, "s": 100},
           data=crash_data)
plt.xlabel('Speeding', size=8)
plt.ylabel('No previous accidents', size=8)
plt.title("Relation between speeding and not previous\n", fontsize=10)
plt.show()

# correlation between ins_premium and ins_losses (0,62)
sns.lmplot(x='ins_losses', y='ins_premium', fit_reg=False, scatter_kws={"color": "blue", "alpha": 0.5, "s": 100},
           data=crash_data)
plt.xlabel('Insurance losses', size=8)
plt.ylabel('Premium insurance', size=8)
plt.title("Relation between Premium insurance and Insurance losses\n", fontsize=10)
plt.show()

# alcohol consumption per state
plt.figure(figsize=(10, 9))
sns.scatterplot(y="abbrev", x="alcohol", data=crash_data, color="red")
plt.ylabel("\nState", fontsize=15)
plt.xlabel("Alcohol", fontsize=15)
plt.title("Alcohol collisions per state\n", fontsize=17)
plt.show()

# analyzing number of drivers first time in accident per state
plt.figure(figsize=(10, 9))
sns.scatterplot(x="no_previous", y="abbrev", data=crash_data, color="red")
plt.xlabel("\nPercentage of drivers involved in fatal collisions first time", fontsize=15)
plt.ylabel("State", fontsize=15)
plt.title("First time collision per state\n", fontsize=17)
plt.show()

# Non-distracted drivers' analysis
sns.violinplot(crash_data['not_distracted'])
plt.xlabel("\nPercentage of drivers involved in fatal collisions who were not distracted", fontsize=10)
plt.title("How safe are you even when you follow the rules?\n", fontsize=10)
plt.show()


def top_10(df, features):
    for f in features:
        l = df[["abbrev", f]].sort_values(by=f, ascending=False).head(10)
        plt.figure(figsize=(6, 3))
        sns.barplot(y="abbrev", x=f, data=l)
        plt.title("Top 10 - " + f)
        plt.show()


top_10(crash_data, cont)
