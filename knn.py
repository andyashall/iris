import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

preditions = [val for val in y_pred]

accuracy = accuracy_score(y_test, preditions)

print(f'Acc: {accuracy}')