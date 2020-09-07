import pandas as pd
import numpy as np

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])

data_num = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312, 'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860, 'Illinois': 12882135})
states = pd.DataFrame({'area': area, 'pop': pop})
states['density'] = states['pop'] / states['area']

# operatons
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))

df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])

area = pd.Series({'New York': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127, 'Alaska': 239651127}, name='population')

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])

dfA = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                   columns=list('AD'))

dfB = pd.DataFrame(rng.randint(0, 20, (3, 3)),
                   columns=list('BAC'))
arrA = np.random.randint(1, 10, (3, 4))
df = pd.DataFrame(arrA, columns=list('QRST'))

# missing values
vals1 = np.array([1, None, 3, 4])
vals2 = pd.Series([1, np.nan, 'hello', None])

df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])

df.fillna