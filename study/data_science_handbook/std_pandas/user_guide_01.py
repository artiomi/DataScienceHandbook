#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:45:08 2020

@author: artiom
"""
import numpy as np
import pandas as pd
from functools import partial

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

s = pd.Series([1, 3, 5, np.nan, 6, 8])
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df2 = pd.DataFrame({'A': 1.,
                       'B': pd.Timestamp('20130102'),
                       'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                        'D': np.array([3] * 4, dtype='int32'),
                        'E': pd.Categorical(["test", "train", "test", "train"]),
                        'F': 'foo'})

df3 = df.copy()
df3['E'] = ['one', 'one', 'two', 'three', 'four', 'three']

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(-2)

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                       'foo', 'foo', 'qux', 'qux'],
                       ['one', 'two', 'one', 'two',
                      'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[0:4]
stacked = df2.stack()
stacked.unstack(level=0)
stacked.unstack(level=1)
stacked.unstack(level=2)

#time study
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = ((prng.asfreq('M', 's') + 1).asfreq('H', 's') + 9).asfreq('T','s')+10

#categorical
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],"raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium","good", "very good"])
df.sort_values(by="grade")
df.groupby("raw_grade").size()


pd.DataFrame.from_dict(dict([('A', [1, 2, 3]), ('B', [4, 5, 6])]),orient='index', columns=['one', 'two', 'three'])

#broadcasting
df = pd.DataFrame({
        'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
        'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
        'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})

dfmi = df.copy()

dfmi.index = pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'),
                                         (1, 'c'), (2, 'a')],
                                        names=['first', 'second'])
   
df1 = pd.DataFrame({'A': [1., np.nan, 3., 5., np.nan],
                    'B': [np.nan, 2., 3., np.nan, 6.]})

df2 = pd.DataFrame({'A': [5., 2., 4., np.nan, 3., 7.],
                       'B': [np.nan, np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)
df1.combine(df2, func=lambda x,y:np.where(pd.isna(x),y,x))


def extract_city_name(df):
    """
    Chicago, IL -> Chicago for city_name column
    """
    df['city_name'] = df['city_and_code'].str.split(",").str.get(0)
    return df

def add_country_name(df, country_name=None):
   """
   Chicago -> Chicago-US for city_name column
   """
   col = 'city_name'
   df['city_and_country'] = df[col] + country_name
   return df

df_p = pd.DataFrame({'city_and_code': ['Chicago, IL']})

df_p.pipe(extract_city_name).pipe(add_country_name, country_name="US")


#Apply function
tsdf = pd.DataFrame(np.random.randn(1000, 3), columns=['A', 'B', 'C'],index=pd.date_range('1/1/2000', periods=1000))

def subtract_and_divide(x, sub, divide=1):
    print('sub:', sub, 'divide:', divide)
    return (x - sub) / divide

df.apply(subtract_and_divide, args=(5,), divide=3)

#custom describe
q_25 = partial(pd.Series.quantile, q=0.25)
q_25.__name__ = '25%'
q_75 = partial(pd.Series.quantile, q=0.75)
q_75.__name__ = '75%'
tsdf.agg(['count', 'mean', 'std', 'min', q_25, 'median', q_75, 'max'])


#transform
tsdf = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],index=pd.date_range('1/1/2000', periods=10))
tsdf.transform([np.exp, np.sqrt])


#read subtypes
def subdtypes(dtype):
   subs = dtype.__subclasses__()
   if not subs:
       return dtype
   return [dtype, [subdtypes(dt) for dt in subs]]












