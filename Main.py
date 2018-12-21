from math import ceil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

# Wczytywanie danych z pliku Adult.csv
original_data = pd.read_csv(
    'Dane/Adult.csv',
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Income"],
        sep=',',
        engine='python',
        na_values="?")

print(original_data.head())


# Histogram dla poszczegolnych kolumn
fig = plt.figure(figsize=(20,15))
cols = 5
rows = ceil(float(original_data.shape[1]) / cols)
for i, column in enumerate(original_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if original_data.dtypes[column] == np.object:
        original_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        original_data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()