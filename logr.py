import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import the data
train = pd.read_csv('./data/iris.csv')

print(train.head())

# Proccess data
train['Species'] = train['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Split the data
x = train.drop('Species', 1)
y = train['Species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.75, random_state=1001)

# Build and fit model
model = LogisticRegression( 
  penalty='l2', 
  dual=False, 
  tol=0.000001, 
  C=10.0,
  fit_intercept=True, 
  intercept_scaling=1, 
  class_weight=None, 
  random_state=1, 
  solver='newton-cg', 
  max_iter=100, 
  multi_class='multinomial', 
  verbose=0, 
  warm_start=False, 
  n_jobs=1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

preditions = [round(val) for val in y_pred]

accuracy = accuracy_score(y_test, preditions)

print(f'Acc: {accuracy}')