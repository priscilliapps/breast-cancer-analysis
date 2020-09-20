# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 22:58:05 2020

@author: prisc
"""

#1 IMPORT AND PREPARE THE DATA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

data = pd.read_csv('data.csv')
#print (data.head())
#print (data.columns)

y = data.diagnosis #storing diagnosis data
cols_to_drop = ['Unnamed: 32', 'id', 'diagnosis']
x = data.drop(cols_to_drop, axis = 1)
print (y.head())
print (x.head())

#2 PLOT THE DATA 
ax = sns.countplot(y, label="Count")
B, M = y.value_counts()
print ('Benign Tumors: ', B)
print ('Malignant Tumors: ', M)

print (x.describe())

#3 STANDARDIZING THE DATA AND VISUALIZE TO SELECT THE FEATURES
data = x
standardized_data = (data - data.mean()) / data.std() #to standardized the data
data = pd.concat([y, standardized_data.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars='diagnosis',
               var_name='features',
               value_name='value')
plt.figure(figsize = (10, 10))
sns.violinplot(x = 'features', 
               y = 'value', 
               hue = 'diagnosis', 
               data = data, 
               split = True, 
               inner = 'quart') #good for visualizing distributions
plt.xticks(rotation = 90)
plt.show()

sns.boxplot(x = 'features', 
            y = 'value', 
            hue = 'diagnosis', 
            data = data)
plt.xticks(rotation = '90')
plt.show()

#sns.jointplot(x.loc[:, 'concavity_mean'], 
#              x.loc[:, 'concave points_mean'], 
#              kind = 'regg',) #to compare two features if they're correlated

sns.set(style = 'whitegrid', palette = 'muted')
sns.swarmplot(x = 'features', 
               y = 'value', 
               hue = 'diagnosis', 
               data = data)
plt.xticks(rotation = 90)
plt.show()

f, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(x.corr(), 
            annot = True, 
            linewidth = .5, 
            fmt = '.1f', 
            ax = ax)
