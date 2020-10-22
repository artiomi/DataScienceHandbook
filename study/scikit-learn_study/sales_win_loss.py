import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

ROOT = '/work/workspaces/pycharm/training_data_science/data/'
sales_data = pd.read_csv(ROOT + 'WA_Fn-UseC_-Sales-Win-Loss.csv')

sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize': (11.7, 8.27)})

# vizualization with countplot
sns.countplot('Route To Market', data=sales_data, hue='Opportunity Result')
sns.despine(offset=10, trim=True)

# vizualization with violinplot
sns.violinplot(x="Opportunity Result", y="Client Size By Revenue", hue="Opportunity Result", data=sales_data)

# print unique values of columns which require to be transformed
print("Supplies Subgroup' : ", sales_data['Supplies Subgroup'].unique())
print("Region : ", sales_data['Region'].unique())
print("Route To Market : ", sales_data['Route To Market'].unique())
print("Opportunity Result : ", sales_data['Opportunity Result'].unique())
print("Competitor Type : ", sales_data['Competitor Type'].unique())
print("'Supplies Group : ", sales_data['Supplies Group'].unique())

# convert string columns to number
le = preprocessing.LabelEncoder()

sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

# define target and data datasets
# select columns other than 'Opportunity Number','Opportunity Result'cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
# dropping the 'Opportunity Number'and 'Opportunity Result' columns

cols = [col for col in sales_data.columns if col not in ['Opportunity Number', 'Opportunity Result']]
data = sales_data[cols]
# assigning the Oppurtunity Result column as target
target = sales_data['Opportunity Result']

# split data into training and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, random_state=10)

# apply Gaussian Naive Bayes algorithm
gnb = GaussianNB()
# train the algorithm on training data and predict using the testing data
fit = gnb.fit(data_train, target_train)
pred = fit.predict(data_test)
print(pred.tolist())
# print the accuracy score of the model
print("Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize=True))

# apply LinearSVC

svc_model = LinearSVC(random_state=0, dual=False)
# train the algorithm on training data and predict using the testing data
fit = svc_model.fit(data_train, target_train)
pred = fit.predict(data_test)
# print the accuracy score of the model
print("LinearSVC accuracy : ", accuracy_score(target_test, pred, normalize=True))

# K-Neighbourhoods classifier
neigh = KNeighborsClassifier(n_neighbors=3)
# Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print("KNeighbors accuracy score : ", accuracy_score(target_test, pred))

# visualize GaussianNB
visualizer = ClassificationReport(gnb, classes=['Won', 'Loss'])
visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.show()  # Draw/show/poof the data

# visualize LinearSVC
visualizer = ClassificationReport(svc_model, classes=['Won', 'Loss'])
visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
visualizer.score(data_test, target_test)  # Evaluate the model on the test data
g = visualizer.show()

# visualization K-Neighbourhoods
visualizer = ClassificationReport(neigh, classes=['Won','Loss'])
visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
visualizer.score(data_test, target_test) # Evaluate the model on the test data
g = visualizer.show()
