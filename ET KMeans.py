
"""
Dr. Etienne Pienaar's code for K-Means clustering, originally written in R but
converted Python, using ChatGPT, then ammended by KM for accuracy 

"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulate 3 concentric circle data
def sim(N=500):
    r = np.random.choice([2, 4, 6], N, replace=True) + np.random.normal(0, 0.1, N)
    X = np.random.uniform(-1, 1, N)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
    return x, y, r

dat_X1, dat_X2, dat_r = sim()

# Data in X-space
X = np.column_stack((dat_X1, dat_X2))

# Distance matrix calculation
def dist_mat(X):
    S = cdist(X, X, 'euclidean') # finds euclidean
    return S

res_S = dist_mat(X) 

# Affinity matrix and Laplacian construction
sig = 0.2
A = np.exp(-res_S**2 / (2 * sig**2))
D = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
L = D @ A @ D

# Eigen decomposition
res2_vals, res2_vecs = eigh(L)
idx = res2_vals.argsort()[::-1]
sorted_eigenvalues = res2_vals[idx]
sorted_eigenvectors = res2_vecs[:, idx]
k = 3
V = sorted_eigenvectors[:, :k]
Vnorm = V


# Normalization function
def renorm(x):
    return x / np.sqrt(np.sum(x**2))

Vnorm = np.apply_along_axis(renorm, 1, V)

# K-means clustering
def k_means(X, initial_index, iterations):
    Mus = X[initial_index, :]
    N, d = X.shape
    K = len(initial_index)
    error = []

    for i in range(1, iterations):
        dists = np.zeros((N, K))
        for j in range(K):
            dists[:, j] = np.sum((X - Mus[j, :])**2, axis=1)

        assigned_labels = np.argmin(dists, axis=1)
        error.append(np.sum(np.min(dists, axis=1)))

        for k in range(K):
            if np.any(assigned_labels == k):
                Mus[k, :] = np.mean(X[assigned_labels == k, :], axis=0)
            else:
                wh_rep = np.argmax(np.max(dists, axis=1))
                Mus[k, :] = X[wh_rep, :]

    return Mus, assigned_labels, dists, K, error

# Picking good starting values for K-means
DV_S = dist_mat(Vnorm)
i1 = 0
i2 = np.argmax(DV_S[i1, :])
i3 = np.argmax(DV_S[i1, :] * DV_S[i2, :])

# Clustering on the chosen eigenvectors
res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(Vnorm, [i1, i2, i3], 20)

# Plot the original data, assigned clusters, and eigenvectors with cluster centroids
fig, axs = plt.subplots(2, 2, figsize=(12, 12)) # creating the figure to be 2x2

# Plot 1
axs[0, 0].scatter(dat_X1, dat_X2, c='black', s=16) # data points on top left figure (Plot 1)
axs[0, 0].set_title('Observations') # Plot 1 title 

# Plot 2
axs[0, 1].scatter(X[:, 0], X[:, 1], c=res_clust_labels + 1, s=16)
axs[0, 1].set_title('Assigned Labels')

fig3d = plt.figure(figsize=(6, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
N = Vnorm.shape[0]
jitter = 0.1 # Perturb observations to make the `clumps' visible. 
ax3d.scatter(Vnorm[:, 0] + np.random.uniform(0, jitter, N), Vnorm[:, 1] + np.random.uniform(0, jitter, N), Vnorm[:, 2] + np.random.uniform(0, jitter, N), c=res_clust_labels + 1, s=16)
ax3d.scatter(res_clust_Mus[:, 0], res_clust_Mus[:, 1], res_clust_Mus[:, 2], c='black', marker='x', s=100)
ax3d.set_title('Eigen Vectors')
plt.show()

# Cluster dispersion
M = 50
sigmas = np.linspace(0.05, 2, M)
cluster_dispersion = np.zeros(M)

for i, sig in enumerate(sigmas):
    A = np.exp(-sim_matrix**2 / (2 * sig**2))
    D = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    L = D @ A @ D
    
    res2_vals, res2_vecs = eigh(L)
    res2_vals = res2_vals[::-1]
    res2_vecs = res2_vecs[::-1]
    V = res2_vecs[:, :-k]
    Vnorm = np.apply_along_axis(renorm, 1, V)
    
    DV_S = dist_mat(Vnorm)
    i1 = 0
    i2 = np.argmax(DV_S[i1, :])
    i3 = np.argmax(DV_S[i1, :] * DV_S[i2, :])
    
    res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(Vnorm, [i1, i2, i3], 20)
    cluster_dispersion[i] = res_clust_error[-1]

plt.figure()
plt.plot(sigmas, cluster_dispersion, 'bo-')
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel('Sigma')
plt.ylabel('Cluster Dispersion')
plt.title('Cluster Dispersion vs Sigma')
plt.show()

# Running algorithm for a set of sigmas
fig, axs = plt.subplots(3, 2, figsize=(12, 18))
sigmas = [0.05, 0.2, 1]
for i, sig in enumerate(sigmas):
    A = np.exp(-res_S**2 / (2 * sig**2))
    D = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    L = D @ A @ D
    
    '''res2_vals, res2_vecs = np.linalg.eigh(L)
    res2_vals = res2_vals[::-1]
    res2_vecs = res2_vecs[::-1]
    V = res2_vecs[:, :k]
    Vnorm = np.apply_along_axis(renorm, 1, V)'''
    
    
    # Eigen decomposition
    res2_vals, res2_vecs = eigh(L)
    idx = res2_vals.argsort()[::-1]
    sorted_eigenvalues = res2_vals[idx]
    sorted_eigenvectors = res2_vecs[:, idx]
    k = 3
    V = sorted_eigenvectors[:, :k]
    Vnorm = V
    
    DV_S = dist_mat(Vnorm)
    i1 = 0
    i2 = np.argmax(DV_S[i1, :])
    i3 = np.argmax(DV_S[i1, :] * DV_S[i2, :])
    
    res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(Vnorm, [i1, i2, i3], 20)
    
    
    N = Vnorm.shape[0]
    jitter = 0.1 # Perturb observations to make the `clumps' visible. 

    axs[i, 0].scatter(X[:, 0], X[:, 1], c=res_clust_labels + 1, s=16)
    axs[i, 0].set_title(f'Sigma = {sig}')
    
    ax3d = fig.add_subplot(3, 2, 2*i + 2, projection='3d')
    ax3d.scatter(Vnorm[:, 0] + np.random.uniform(0, jitter, N), Vnorm[:, 1] + np.random.uniform(0, jitter, N), Vnorm[:, 2] + np.random.uniform(0, jitter, N), c=res_clust_labels + 1, s=16)
    ax3d.scatter(res_clust_Mus[:, 0], res_clust_Mus[:, 1], res_clust_Mus[:, 2], c='black', marker='x', s=100)
    ax3d.set_title('Eigen Vectors')

plt.show()
