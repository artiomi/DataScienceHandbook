import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
ROOT = '/work/workspaces/pycharm/training_data_science/data/'

df = pd.read_csv(ROOT + 'drug-use-by-age.csv')
# 1 check missing data
print(df.info())
# there are some columns with object data type, they are expected to be numbers
# for missing values is used '-'
# mask Series with '-'
print(df[df.isin(['-'])].any())

# replace '-' with Nan, then replace Nan with mean
# get list of columns

cols = list(df.columns)
for col in cols:
    print('Check for - in:', col)
    if df[col].isin(['-']).any():
        print('Replacing - with nan')
        df.loc[df[col].isin(['-']), col] = 0.0
        df[col] = df[col].astype(np.float64)
        print('Replacement finished')

print(df.info())
print(df.isna().any())
# replace NA with mean
# for col in cols:
#     if df[col].isna().any():
#         print('Fill na on column', col)
#         df[col].fillna(round(df[col].mean(), 1), inplace=True)
print(df[['age', 'n', 'alcohol-use', 'alcohol-frequency']].head())

# alcohol consumption per age
plt.figure(figsize=(10, 9))
sns.scatterplot(x="alcohol-use", y="age", data=df, color="red")
plt.xlabel("Alcohol", fontsize=15)
plt.ylabel("\nAge", fontsize=15)
plt.title("Alcohol per age\n", fontsize=17)
plt.show()

# how is alcohol consumed per age
plt.figure(figsize=(10, 9))
sns.scatterplot(x='alcohol-frequency', y="age", data=df, color="red")
plt.xlabel("Alcohol frequency", fontsize=15)
plt.ylabel("\nAge", fontsize=15)
plt.title("Alcohol frequency in last 12 month per age\n", fontsize=17)
plt.show()

l = df[['age', "heroin-frequency"]].sort_values(by="age", ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(y="age", x="heroin-frequency", data=l)
plt.title = "heroin frequency per age"
plt.show()

# transform age to range index
# df.loc[(df['age'].str.len() > 3), 'age'] = df.loc[(df['age'].str.len() > 3), 'age'].str[3:]
# df.loc[16:'age', 'age'] = 150
# df['age'] = df['age'].astype(np.int)
# ages = list(df['age'])
# ages.insert(0, 0)
# index = pd.IntervalIndex.from_breaks(ages)
# df.index = index
# df.reindex()
# df.drop('age', axis=1, inplace=True)
# print(df.head())


# plot of median usage per age and percentage of those who tried
indexes = [col.rsplit('-', 1)[0] for col in df.columns[2::2]]
for ind in df.index:
    age = df.loc[ind, 'age']
    number = df.loc[ind, 'n']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    median, percentage = df.iloc[ind, 3::2], df.iloc[ind, 2::2]
    median.index = indexes
    percentage.index = indexes
    median.plot(ax=ax1, kind='bar', colormap='Accent',
                xlabel="Drugs", ylabel="Median",
                title=f"Median number of times a user of age:{age}.Total: {number}")
    percentage.plot(ax=ax2, kind='bar', xlabel="Drugs", ylabel="Usage Percentage",
                    title=f"Percentage of those at age :{age}.Total : {number}")
    plt.title = f"Total analyzed {number} persons."
    plt.show()

corr = df.iloc[:, 2:].corr()
plt.figure(figsize=(21, 14))
# plt.title("The Correlations of the Features")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, mask=mask, cmap="inferno")
plt.show()
# TO DO it would be nice to find relationship between age, drug use and usage frequency
