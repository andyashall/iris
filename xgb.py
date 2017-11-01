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

