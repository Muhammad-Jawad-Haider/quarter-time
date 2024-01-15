import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from math import sqrt


# Import data
df = pd.read_csv('mtcars.csv')

# Split data into test train
X = df.drop(columns=['model', 'qsec'])
y = df['qsec']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train decision tree and calculate mean square error
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

error = sqrt(((y_pred - y_test)**2).mean())
