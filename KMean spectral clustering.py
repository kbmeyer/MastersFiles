#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:15:50 2024

@author: kirkmeyer

Here we seek to embed the Kmeans clustering algo into spectral clustering


"""

# Required Packages 

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sklearn.metrics as metric
from Function_File import alc, clus_lc, k_means, cdist, renorm
from scipy.spatial.distance import cdist
import scipy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
from tqmd import tqmd

"""
                               #### Datasets ####

"""

# Dataset 1 - Blobs 

def KM_makeblobs(N, num_clusters):
    blobs,labels = make_blobs(n_samples=N,
                             n_features=5000,
                             shuffle=False,
                             centers=num_clusters,
                             random_state=0)
    
    blob_plot = plt.scatter(blobs[:, 0], blobs[:, 1], c=labels, s=50, cmap='viridis')
    
    return blob_plot, blobs, labels


test_blob = KM_makeblobs(500, num_clusters = 3)


# Dataset 2 - 3 Concentric Cirle 

def circle_sim(N):
    label = np.random.choice([1, 2, 3], N, replace=True)
    r = 2*label + np.random.normal(0, 0.1, N)
    X = np.random.uniform(-1, 1, N)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
    
    return x, y, r, label

def KM_makecircle(N):
    
    dat_X1, dat_X2, dat_r, dat_label = circle_sim(N)
    
    X = np.column_stack((dat_X1, dat_X2))
    
    cir_plot = plt.scatter(dat_X1, dat_X2, c = dat_label,  s = 50, cmap='viridis')
    
    return cir_plot, dat_label, X


test_circle = KM_makecircle(500)

def dist_mat(X):
    S = cdist(X, X, 'euclidean') # finds euclidean distance
    return S
    
       


"""
####---------------------- Spectral Decomposition  ------------------------####

"""

# Setting up the Laplacians

def KM_spec_decom(sim_mat, sigma, Laplacian, I_mat):
    
    
    A = np.exp(-sim_mat**2 / (2 * sigma**2)) # Gaussian Affinity matrix


    if Laplacian == "RW":
        
        D_1     = np.diag(1/np.sum(A,axis = 1)) #D^{-1}
        L_rw    = I_mat  - D_1 @ A
        eig_vals, eig_vecs  = la.eigh(L_rw) # RW eigendecomposition 
    
    if Laplacian == "SYM":
        
        D_2     = np.diag(1 / np.sqrt(np.sum(A, axis=1))) #D^{-1/2}
        L_sym   = I_mat - D_2 @ A @ D_2 
        eig_vals, eig_vecs_sym = la.eigh(L_sym) # SYM eigendecomposition 
        
        eig_vecs = renorm(eig_vecs_sym)
        
        
        
    return eig_vals, eig_vecs


"""
#### --------------------------------------------------------------------- ####

"""



sigma   = 0.2
sim_mat = dist_mat(test_circle[2]) 
I       = np.eye(len(sim_mat))
A       = np.exp(-sim_mat**2 / (2 * sigma**2)) # Gaussian Affinity matrix



######################## Eigenvector comparison ###############################







dot_mat_rw = np.zeros((500,500))

for i in range(len(A)):
    
    vect_1 = eig_vecs_rw[:,i]
    
    for j in range(len(A)):
        
        vect_2 = eig_vecs_rw[:,j]
        
        dot_mat_rw[i,j] = np.dot(vect_1, vect_2)
        
sns.heatmap(dot_mat_rw, vmin = np.min(dot_mat_rw), vmax = np.max(dot_mat_rw))

#-----------------------------------------------------------------------------#

dot_mat_sym = np.zeros((500,500))

for i in range(len(A)):
    
    vect_1 = eig_vecs_sym[:,i]
    
    for j in range(len(A)):
        
        vect_2 = eig_vecs_sym[:,j]
        
        dot_mat_sym[i,j] = np.dot(vect_1, vect_2)
        
sns.heatmap(dot_mat_sym, vmin = np.min(dot_mat_sym), 
            vmax = np.max(dot_mat_sym))

#-----------------------------------------------------------------------------#

test_eig = eig_vecs_rw[0:10,0:10]

test_dot = np.zeros((len(test_eig),len(test_eig)))

for i in range(len(test_eig)):
    
    vect_1 = test_eig[:,i]
    
    for j in range(len(test_eig)):
        
        vect_2 = test_eig[:,j]
        
        test_dot[i,j] = np.dot(vect_1, vect_2)

sns.heatmap(test_dot, vmin = np.min(test_dot), vmax = np.max(test_dot))

test_selfdot = np.dot(test_eig[:,2], test_eig[:,1])

#-----------------------------------------------------------------------------#

########################## Selection Criteria ################################

''' 

With us using Vanilla K-Means on the spectral decomposed data, we need to tune
for 2 hyper parameters k (number of clusters) and sigma of the gaussian kernel
("neighbourhood size" control).  

'''

M = 50
sigmas = np.linspace(0.05, 2, M)
sim_matrix = dist_mat(test_circle[2])
I = np.eye(len(sim_matrix))


# Silhouette Score



# ARI

for sig in sigmas:
    
    val, vec = KM_spec_decom(sim_matrix, sigma = sig, Laplacian = "SYM", I_mat = I)
    ALC_spec_GM = alc(np.corrcoef(Vnorm),"GM")
    num_clusters = np.max(ALC_spec_GM[0])
    

# Mutual Information

# Calinski-Harabasz Index

# Davies-Bouldin Index



############################## Clustering Run #################################






################################  Results  #################################### 

