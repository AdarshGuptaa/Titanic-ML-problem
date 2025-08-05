import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

# Changes style and look of the plotted graphs
plt.style.use('ggplot')

# Loads csv file
df = pd.read_csv('train.csv')

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

df['Family_Count'] = df['SibSp'] + df['Parch']

# prefixes = set(df['Cabin'].str[:1].tolist())
# print(prefixes)

# Column datatype change
# df['Survived'] = df['Survived'].astype(bool)

# Check null values
# print(df.isna().sum())

# Check duplicates
# print(df.loc[df.duplicated()])
# print(df.loc[df.duplicated(subset = ['PassengerId'])])

# Plot analysis
# plt.bar(df['Pclass'], df['Fare'] )
# # plt.yticks([0, 1], ['False', 'True'])
# plt.grid(True)
# plt.show()


print(df.head(5))


