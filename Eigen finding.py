#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 13:14:45 2024

@author: kirkmeyer
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from numba import jit
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.linalg as la 
from sklearn import metrics
import seaborn as sns

A = [
     [33, 24], 
     [48, 47],
]


D = np.diag(np.sum(A, axis = 0))

vals,V = np.linalg.eigh(A)
    
    



















