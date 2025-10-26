import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Create non-linear data
X = np.linspace(0,10,100).reshape(-1,1)
y = np.sin(X).ravel() + np.random.randn(100)*0.1

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#High bias model
lr = LinearRegression().fit(X_train, y_train)

#High variance model
dt = DecisionTreeRegressor(max_depth=None).fit(X_train, y_train)

#compare both
print("Linear Regression Test Error:", mean_squared_error(y_test, lr.predict(X_test)))
print("Decision Tree Test Error:", mean_squared_error(y_test, dt.predict(X_test)))