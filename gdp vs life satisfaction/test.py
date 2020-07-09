# Some discrepancies here. See lin_reg_model model only

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

# Load the data
# Prepare the data
country_stats = pd.read_csv("datasets_894_813759_2015.csv")
X = np.c_[country_stats["Economy (GDP per Capita)"],country_stats["Trust (Government Corruption)"]]
y = np.c_[country_stats["Happiness Score"]]


print(country_stats.head())

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Accuracy: " + str(acc))


# Select a linear model
lin_reg_model = sklearn.linear_model.LinearRegression()
# Train the model
lin_reg_model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[1.32548,0.48357]] # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new)) # outputs [[ 5.96242338]]




# Visualize the data
country_stats.plot(kind='scatter', x="Economy (GDP per Capita)", y='Happiness Score')
plt.show()