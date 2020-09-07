import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

ROOT = '/work/workspaces/pycharm/training_data_science/data/'

# #planets
# pop = pd.read_csv(ROOT + 'state-population.csv')
# areas = pd.read_csv(ROOT + 'state-areas.csv')
# abbrevs = pd.read_csv(ROOT + 'state-abbrevs.csv')
#
# # birthdates
# births = pd.read_csv(ROOT + 'births.csv')

# def read_from_json(file_name):
#     try:
#         recipes = pd.read_json(ROOT + file_name)
#         return recipes
#     except ValueError as e:
#         print("ValueError:", e)
#
#
# recipes = read_from_json('recipeitems-latest.json')
# with open(ROOT + 'recipeitems-latest.json') as f:
#     # Extract each line
#     data = [line.strip() for line in f]
#     # Reformat so each line is the element of a list
#     data_json = "[{0}]".format(','.join(data))
#     # read the result as a JSON
# # read the result as a JSON
# recipes = pd.read_json(data_json.encode())

# seatle bridge

data = pd.read_csv(ROOT + 'Fremont_Bridge_Bicycle_Counter.csv', index_col='Date', parse_dates=True)
data.columns = ['Total','East','West']
