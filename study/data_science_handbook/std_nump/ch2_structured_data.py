import numpy as np

data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'), 'formats': ('U10', 'i4', 'f8')})
print(data.dtype)
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
print("type:", tp)
X = np.zeros(1, dtype=tp)
print(X['mat'][0].shape)

data_rec = data.view(np.recarray)
