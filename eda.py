import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

# Changes style and look of the plotted graphs
plt.style.use('ggplot')

# Loads csv file
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Sets max visible number of columns 
pd.set_option('display.max_columns', 200)


# Total data(Rows, Columns) 
# print(df.shape)

# List all Column names
# print(df.columns)

# List the data type of each column
# print(df.dtypes)

# Returns some statistical data on the dataset
# print(df.describe())

# Dropping useless features (Cabin has incomplete data and Name is irrelevant)
df = df[['PassengerId', 'Survived', 'Pclass',
        #  'Name',
         'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 
        # 'Cabin',
        'Embarked']].copy()

df_test = df_test[['PassengerId', 'Pclass',
        #  'Name',
         'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 
        # 'Cabin',
        'Embarked']].copy()

df['Family_Count'] = df['SibSp'] + df['Parch']

df_test['Family_Count'] = df['SibSp'] + df['Parch']

df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})

df_test['Sex'] = df_test['Sex'].replace({'male' : 1, 'female': 0})

# For Label Encoding and Preserving Priority order
# Remap the values
remap = {1: 3, 2: 2, 3: 1}

df['Pclass'] = df['Pclass'].map(remap)

df_test['Pclass'] = df_test['Pclass'].map(remap)

# Replace NaN in 'Age' with average (mean) age
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

df_test['Age'] = df['Age'].fillna(mean_age)

mean_fare = df['Fare'].mean()
df_test['Fare'] = df_test['Fare'].fillna(mean_fare)

df = pd.get_dummies(df, columns=['Embarked'], drop_first = True)
df_test = pd.get_dummies(df_test, columns=['Embarked'], drop_first = True)

df['Embarked_Q'] = df['Embarked_Q'].astype(int)
df['Embarked_S'] = df['Embarked_S'].astype(int)

df_test['Embarked_Q'] = df_test['Embarked_Q'].astype(int)
df_test['Embarked_S'] = df_test['Embarked_S'].astype(int)

# prefixes = set(df['Cabin'].str[:1].tolist())
# print(prefixes)

# Column datatype change
df['Survived'] = df['Survived'].astype(int)

# Check null values
# print(df_test.isna().sum())

# Check duplicates
# print(df.loc[df.duplicated()])
# print(df.loc[df.duplicated(subset = ['PassengerId'])])

# Plot analysis
# plt.bar(df['Pclass'], df['Fare'] )
# # plt.yticks([0, 1], ['False', 'True'])
# plt.grid(True)
# plt.show()

# Export to CSV
df.to_csv('updated_train.csv', index=False)
df_test.to_csv('updated_test.csv', index = False)

# print(df.head(5))




