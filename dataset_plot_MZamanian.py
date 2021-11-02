# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% Assignment 1 - Visualize Iris Dataset
# reference: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html

# %% part 0 - import necessary modules
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# %% part 1 - load the data
iris = load_iris()
fig_w = 7; fig_h = fig_w / 1.618

# %% part 2 - perform EDA (visualize the data)
# EDA = Exploratory Data Analysis

# %% section 1 - Sepal plot
# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the color-bar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)].capitalize())

plt.figure(figsize=(fig_w, fig_h))
plt.scatter(iris.data[:, x_index],
            iris.data[:, y_index],
            c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index].capitalize())
plt.ylabel(iris.feature_names[y_index].capitalize())

plt.tight_layout()
# plt.grid()
plt.savefig('iris-dataset-sepal.png')
plt.show()

# %% section 2 - Petal plot
# The indices of the features that we are plotting
x_index = 2
y_index = 3

# this formatter will label the color-bar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)].capitalize())

plt.figure(figsize=(fig_w, fig_h))
plt.scatter(iris.data[:, x_index],
            iris.data[:, y_index],
            c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index].capitalize())
plt.ylabel(iris.feature_names[y_index].capitalize())

plt.tight_layout()
# plt.grid()
plt.savefig('iris-dataset-petal.png')
plt.show()

# %%
