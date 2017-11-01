import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Import the data
train = pd.read_csv('./data/iris.csv')

print(train.head())

# Proccess data
train['Species'] = train['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Split the data
x = train.drop(['Species', 'Id'], 1)
y = train['Species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# feature_columns = []

# for col in list(X_train):
#   feature_columns.append(tf.feature_column.numeric_column(col, shape=[1]))

# Build and fit model
model = tf.estimator.DNNClassifier(
  hidden_units  = [10, 20, 10],
  feature_columns=feature_columns,
  n_classes=3,
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_train.as_matrix())},
    y=np.array(y_train),
    num_epochs=None,
    shuffle=True)

model.train(train_input_fn, steps=2000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_test.as_matrix())},
    y=np.array(y_test),
    num_epochs=1,
    shuffle=False)

accuracy_score = model.evaluate(input_fn=test_input_fn)["accuracy"]

print(f'Acc: {accuracy_score}')
