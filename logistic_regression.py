import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Changes style and look of the plotted graphs
plt.style.use('ggplot')

# Loads csv file
df = pd.read_csv('updated_train.csv')
df_test = pd.read_csv("updated_test.csv")

# Sets max visible number of columns 
pd.set_option('display.max_columns', 200)

# Initiate Model
model = LogisticRegression()

x_train = df[['Pclass','Sex','Age','Fare','Family_Count']]
y_train = df['Survived']

x_test = df_test[['Pclass','Sex','Age','Fare','Family_Count']]


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

sol = pd.DataFrame({'PassengerId' : df_test['PassengerId'], 'Survived' : y_pred})

print(sol.shape)

sol.to_csv('logistic_regression_sol.csv', index = False)
# Optional: Feature coefficients
print("Feature Coefficients:", model.coef_)
print("Intercept:", model.intercept_)