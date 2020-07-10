# Loading Data
import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# take all rows and all columns from index 1 up to index 2 but not including index 2 (upper bound range is not included)

# Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Visualize Linear Regression Results
import matplotlib.pyplot as plt

plt.scatter(X,y, color="red")
plt.plot(X, lin_reg.predict(X))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Linear Regression prediction
lin_reg.predict([[6.5]])

# Convert X to polynomial format
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
# For Polynomial Regression, we need to transform our matrix X to X_poly where X_poly will contain X to the power of n — depending upon the degree we choose.
# If we choose degree 2, then X_poly will contain X and X to the power 2.
# If we choose degree 3, then X_poly will contain X, X to the power 2 and X to the power 3.
# We will be using the PolynomialFeatures class from the sklearn.preprocessing library for this purpose. When we create an object of this class — we have to pass the degree parameter.



# Passing X_poly to LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualize Poly Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title("Poly Regression - Degree 4")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# Polynomial Regression prediction
new_salary_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)