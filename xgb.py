import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the data
train = pd.read_csv('./data/iris.csv')

print(train.head())

# Split the data
x = train.drop('Species', 1)
y = train['Species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.75, random_state=1001)

# Prams for classifier
n_estimators = 90
learning_rate = .01
max_depth = 4
subsample = 1
colsample_bytree = 1
gamma = 0
max_delta_step = 0
min_child_weight = 1

# Build and fit model
model = XGBClassifier(n_estimators = n_estimators,
                      max_depth = max_depth,
                      learning_rate = learning_rate,
                      subsample = subsample,
                      colsample_bytree = colsample_bytree,
                      gamma = gamma,
                      max_delta_step = max_delta_step,
                      min_child_weight = min_child_weight
                      )
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

preditions = [val for val in y_pred]

accuracy = accuracy_score(y_test, preditions)

print(f'Acc: {accuracy}')