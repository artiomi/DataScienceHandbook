import numpy as np
import pandas as pd


def scalar_sum(s1, s2):
    print("Summing:", s1, s2)
    return s1 + s2


data = pd.Series([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
data2 = pd.Series([1, 2, 3, 4, 5])
data3 = data.combine(data2, scalar_sum)
print("data3:\n", data3)

print("original data:\n", data)

population_dict = {'California': 38332521, 'Texas': 26448193, 'New York': 19651127, 'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population['California': 'Florida'])

cust_index = pd.Series([5, 6, 7, 8], index=[100, 200, 300, 'a'])
print("custom index:\n", cust_index)

# data frames################
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}

area = pd.Series(area_dict)
states = pd.DataFrame({'population': population, 'area': area})
print("staes DF:\n", states)

names_df = pd.DataFrame(['Jhon', 'Bob', 'Tom'], index=[1, 2, 3], columns=['names'])
print('names df:\n', names_df)

A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
numpy_df = pd.DataFrame(A)
print("numpy DF:\n", numpy_df)

######### PD indexes ###############
ind = pd.Index([2, 3, 5, 7, 11])
