import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
X_train = [[6], [7], [8], [9], [10], [11], [12]] #average shoe size
y_train = [[1.6], [1.7], [1.75], [1.78], [1.85], [1.9], [2]] #average heights (in meters) per shoe size

# Testing set
X_test = [[6], [7.5], [9.5], [13]] #shoe sizes
y_test = [[1.7], [1.7], [1.8], [1.88]] # heights

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Shoe size matched against individual height')
plt.xlabel('Shoe Size (RSA)')
plt.ylabel('Height (in meters)')
plt.scatter(X_train, y_train)
plt.show()



