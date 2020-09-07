import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

titanic.pivot_table('survived', index='sex', columns='class')


def not_survived(x):
    x.loc[x['survived'] == 0, 'not_survived'] = 1
    x.loc[x['survived'] == 1, 'not_survived'] = 0
    return x
