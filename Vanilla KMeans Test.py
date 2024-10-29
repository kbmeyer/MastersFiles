#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:20:03 2024

@author: kirkmeyer
"""


from Function_File import alc, clus_lc, k_means, cdist, renorm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import scipy.linalg as la
import seaborn as sns
import pandas as pd
from openpyxl.workbook import Workbook
import sklearn.metrics as metric
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import random

###############################################################################

# Functions 


def KM_makeblobs(N, num_clusters):
    blobs,labels = make_blobs(n_samples=N,
                             n_features=5000,
                             shuffle=False,
                             centers=num_clusters,
                             random_state=0)
    
    blob_plot = plt.scatter(blobs[:, 0], blobs[:, 1], c=labels, s=50, cmap='viridis')
    
    return blob_plot, blobs, labels

def circle_sim(N):
    label = np.random.choice(3, N, replace=True)
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


def KM_spec_decom(sim_mat, sigma, Laplacian, I_mat):
    
    
    A = np.exp(-1*sim_mat**2 / (2 * sigma**2)) # Gaussian Affinity matrix


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

def dist_mat(X):
    S = cdist(X, X, 'euclidean') # finds euclidean distance
    return S
    
      
# Reading in data 

data = KM_makecircle(500)
sim_matrix = dist_mat(data[2])
I = np.eye(len(sim_matrix))

# Spectral Decomposition 
eig_vals, eig_vecs = KM_spec_decom(sim_mat = sim_matrix, 
                                   sigma = 0.24, Laplacian = "SYM", 
                                   I_mat = I)

# Eigen vector comparison 

dot_mat = np.zeros((500,500))

for i in range(len(eig_vecs)):
    
    vect_1 = eig_vecs[:,i]
    
    for j in range(len(eig_vecs)):
        
        vect_2 = eig_vecs[:,j]
        
        dot_mat[i,j] = np.dot(vect_1, vect_2)
        
sns.heatmap(dot_mat, vmin = np.min(dot_mat), vmax = np.max(dot_mat))

df_vecs = pd.DataFrame(eig_vecs)
df_vals = pd.DataFrame(eig_vals)

df_vecs.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/eigvecs.xlsx", 
                 sheet_name = "Eigenvectors", index = False)

df_vals.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/eigvals.xlsx", 
                 sheet_name = "Eigenvalues", index = False)


# Selection Criteria 

#range_n_clusters = [2, 3, 4, 5, 6]
'''
for n_clusters in range_n_clusters:
'''    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(eig_vecs) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    
    
    initial_idx =np.array([random.randint(0,499) for _ in range(n_clusters)])
    
    res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(eig_vecs, initial_idx, 10000)
    cluster_labels = res_clust_labels

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(eig_vecs, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(eig_vecs, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        eig_vecs[:, 0], eig_vecs[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    #
    
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
#plt.show()

data = KM_makecircle(500)
sim_matrix = dist_mat(data[2])
I = np.eye(len(sim_matrix))
range_n_clusters = [2, 3, 4]
M = 20
sigmas = np.linspace(0.05, 1, M)
ari_results = []
sil_results = []

for sigma in enumerate(sigmas):
    
    eig_vals, eig_vecs = KM_spec_decom(sim_mat = sim_matrix, 
                                       sigma = sigma[1], Laplacian = "SYM", 
                                       I_mat = I)

    for n_clusters in range_n_clusters:
        
  
        
        #XX = eig_vecs[:,:n_clusters]
        
        XX = eig_vecs
        
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(XX) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        
        
        initial_idx =np.array([random.randint(0,499) for _ in range(n_clusters)])
        
        res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(XX, initial_idx, 10000)
        cluster_labels = res_clust_labels
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(XX, cluster_labels)
        ARI = metric.adjusted_rand_score(data[1], cluster_labels)
        
        ari_results.append([sigma[1], n_clusters, ARI])
        sil_results.append([sigma[1], n_clusters, silhouette_avg])
        
        print(
            "For n_clusters =",
            n_clusters,
            "The sigma is :",
            sigma[1],
            "The average silhouette_score is :",
            silhouette_avg,
            "The ARI is",
            ARI
            
        )
        
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(XX, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            data[2][:, 0], data[2][:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )
    
        #
        
        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d and sigma = %.2f"
            % (n_clusters, sigma[1]),
            fontsize=14,
            fontweight="bold",
        ) 

plt.show()


# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(eig_vecs[:,:3], cluster_labels)

y_lower = 10
for i in range(3):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 3)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(
    data[2][:, 0], data[2][:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
)

#

plt.suptitle(
    "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
    % n_clusters,
    fontsize=14,
    fontweight="bold",
) 

plt.show()


# Clustering ALC 

eig_vals, eig_vecs = KM_spec_decom(sim_mat = sim_matrix, 
                                   sigma = 0.4, Laplacian = "SYM", 
                                   I_mat = I)


k = 3
Y = eig_vecs[:, :k]
Y_corr = np.corrcoef(Y)
ALC_sol = alc(G = Y_corr, method = "GM")

plt.scatter(data[2][:,0], data[2][:,1], c = ALC_sol, s = 50, cmap='viridis')

# Clustering K-Means

DV_S = dist_mat(eig_vecs[:, :k])
i1 = 0
i2 = np.argmax(DV_S[i1, :])
i3 = np.argmax(DV_S[i1, :] * DV_S[i2, :])

res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(eig_vecs, [1, 250, 450, 300, 20], 10000)
plt.scatter(data[2][:,0], data[2][:,1], c = res_clust_labels, s = 50, cmap='viridis')









