#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:10:54 2019

@author: samaneh
"""
# import data
from sklearn.datasets import load_digits
digits = load_digits()
x_digits, y_digits = digits.data, digits.target
print(digits.keys())

# plot data
import matplotlib.pyplot as plt
n_row, n_col = 2, 5

def print_digits(images, y, max_n=10):
    fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row)) # set up the figure size in inches
    i = 0
    while i < max_n and i < images.shape[0]:
        p = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone, interpolation='nearest')
        p.text(0, -1, str(y[i])) # label the image with the target value
        i = i + 1
print_digits(digits.images, digits.target, max_n=10)     

# plot transformed data with PCA
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white',
              'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = x_pca[:, 0][y_digits == i]
        py = x_pca[:, 1][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

# using PCA
from sklearn.decomposition import PCA
estimator = PCA(n_components=10)
x_pca = estimator.fit_transform(x_digits)
plot_pca_scatter()

# plot all components
def print_pca_components(images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape((8, 8)), interpolation='nearest')
        plt.text(0, -1, str(i + 1) + '-component')
        plt.xticks(())
        plt.yticks(())

print_pca_components(digits.images, n_col, n_row) 

# plot transformed data with KPCA
def plot_kpca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white',
              'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = x_kpca[:, 0][y_digits == i]
        py = x_kpca[:, 1][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')                  

# using kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
estimator = kpca.fit_transform(x_digits)
x_kpca = kpca.inverse_transform(estimator)
plot_kpca_scatter()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        