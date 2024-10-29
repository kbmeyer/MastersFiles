#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:50:06 2024

@author: kirkmeyer
"""


import numpy as np
import matplotlib.pyplot as plt


def circle_sim(n_points, n_circles):
    label = np.random.choice(range(1,n_circles+1), n_points, replace=True)
    r = 2*label + np.random.normal(0, 0.1, n_points)
    X = np.random.uniform(-1, 1, n_points)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
    
    return x, y, r, label


def KM_makecircle(n_points, n_circles):
    
    label = np.random.choice(range(1,n_circles+1), n_points, replace=True)
    r = 2*label + np.random.normal(0, 0.1, n_points)
    X = np.random.uniform(-1, 1, n_points)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
        
    X = np.column_stack((x, y))
    
    cir_plot = plt.scatter(x, y, c = label,  s = 50, cmap='viridis')
    
    return cir_plot, label, X



circle_plot, dat_label, dat_X = KM_makecircle(500, 5)


