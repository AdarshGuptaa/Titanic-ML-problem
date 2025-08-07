import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

model = DecisionTreeClassifier()

df_train = pd.read_csv('updated_train.csv')

df_test = pd.read_csv('updated_test.csv')

x = df_train[['Pclass','Sex','Age','Fare','Family_Count','Embarked_Q','Embarked_S']]
y = df_train['Survived']

x_test = df_test[['Pclass','Sex','Age','Fare','Family_Count','Embarked_Q','Embarked_S']]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=35)

model.fit(x, y)

y_pred = model.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)

solution = pd.DataFrame({'PassengerId' : df_test['PassengerId'], 'Survived' : y_pred})

solution.to_csv('decision_tree_sol.csv', index=False)