import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

# Changes style and look of the plotted graphs
plt.style.use('ggplot')

# Loads csv file
df = pd.read_csv('test.csv')

# Sets max visible number of columns 
pd.set_option('max_columns', 20)

print(df.shape)


print(df.head(5))