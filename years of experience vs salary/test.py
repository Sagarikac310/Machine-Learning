from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



# Load Data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values
#  take all rows and all columns except last one

print(dataset.head())



# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)



# Fit Simple Linear Regression to Training Data
linear = LinearRegression()
linear.fit(X_train, y_train)



# Make Prediction
y_pred = linear.predict(X_test)



# Visualize training set results
# plot the actual data points of training set
plt.scatter(X_train, y_train, color = 'red')
# plot the regression line
plt.plot(X_train, linear.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# Visualize test set results
# plot the actual data points of test set
plt.scatter(X_test, y_test, color = 'red')
# plot the regression line (same as above)
plt.plot(X_train, linear.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# Make new prediction
new_salary_pred = linear.predict([[15]])
print('The predicted salary of a person with 15 years experience is ',new_salary_pred)
