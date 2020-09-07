import seaborn as sns
import pandas as pd
import numpy as np

planets = sns.load_dataset('planets')
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))

df1 = pd.DataFrame({'A': rng.rand(5), 'B': rng.rand(5)})

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                  columns=['key', 'data1', 'data2'])

L = [0, 1, 0, 1, 2, 0]


def filter_func(x):
    return x['data2'].std() > 4


def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x
