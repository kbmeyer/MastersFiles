#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:35:38 2024

@author: kirkmeyer
"""

import Function_File as FF
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as la
import seaborn as sns
from plotnine import ggplot, aes, geom_line, labs, geom_point
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker




# Data Generation 
blobs_plots, blobs, blob_labels = FF.KM_makeblobs(N = 1500, num_clusters = 7)


random.seed(6)
circle_plots, circle_labels, circles = FF.KM_makecircle(n_points = 1500 , n_circles = 3)


# Blob Data Plot

plt.scatter(blobs[:, 2], blobs[:, 10],
                        c = blob_labels, s=5, 
                        cmap='viridis')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Blobs.pdf")


# Circle Data Plot 


plt.scatter(circles[:,0], circles[:,1], 
            c = circle_labels,  s = 5, cmap='winter')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Circle.pdf")




# Finding the optimal parameter

sim_blobs = FF.dist_mat(blobs)
sim_circle = FF.dist_mat(circles)
I = np.eye(len(sim_blobs))



sil_blob_k, ari_blob_k = FF.KM_cluster_tuning(data = blobs, true_labels = blob_labels, 
                                              n_sigmas = 100, start_sigma = 0.15, 
                                              end_sigma = 60, Laplacian = "SYM", 
                                               range_n_clusters = [7], 
                                              method = "ET K-Means", plot = "False")


sil_blob_alc, ari_blob_alc = FF.KM_cluster_tuning(data = blobs, true_labels = blob_labels, 
                                              n_sigmas = 100, start_sigma = 10, 
                                              end_sigma = 60, Laplacian = "SYM", 
                                              range_n_clusters = [8], 
                                              method = "LY ALC", plot = "False")



sil_cir_k, ari_cir_k = FF.KM_cluster_tuning(data = circles, true_labels = circle_labels, 
                                              n_sigmas = 20, start_sigma = 0.1, 
                                              end_sigma = 2, Laplacian = "SYM", 
                                              range_n_clusters = [2,3,4], 
                                              method = "ET K-Means", plot = "False")


sil_cir_alc, ari_cir_alc = FF.KM_cluster_tuning(data = circles, true_labels = circle_labels, 
                                              n_sigmas = 20, start_sigma = 0.05, 
                                              end_sigma = 2, Laplacian = "SYM", 
                                              range_n_clusters = [2,3,4], 
                                              method = "LY ALC", plot = "False")


'''
ARI_Blobs_KMeans_df = pd.DataFrame(ari_blob_k)
ARI_Blobs_KMeans_df.columns = ["Sigma", "Num. of Clusters", "ARI"]
print(ARI_Blobs_KMeans_df)

ARI_Blobs_KMeans_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/ARI_KMeans_Blobs_large.xlsx", 
                 sheet_name = "Blobs", index = True)

opt_sigma_kmeans = ARI_Blobs_KMeans_df.iloc[ARI_Blobs_KMeans_df.idxmax()["ARI"]]["Sigma"]



ARI_Blobs_ALC_df = pd.DataFrame(ari_blob_alc)
ARI_Blobs_ALC_df.columns = ["Sigma", "Num. of Clusters", "ARI"]
print(ARI_Blobs_ALC_df)


ARI_Blobs_ALC_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/ARI_ALC_Blobs_large.xlsx", 
                 sheet_name = "Blobs", index = True)

opt_sigma_ALC = ARI_Blobs_ALC_df.iloc[ARI_Blobs_ALC_df.idxmax()["ARI"]]["Sigma"]

eig_vals_blob, eig_vecs_blob = FF.KM_spec_decom(sim_mat = sim_blobs, 
                                                sigma = 24.4545, 
                                                Laplacian = "SYM", I_mat = I)

Blobs_eigenvector_df = pd.DataFrame(eig_vecs_blob)
Blobs_eigenvalue_df = pd.DataFrame(eig_vals_blob)

Blobs_eigenvector_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/Blobs_Eigenvector.xlsx", 
                 sheet_name = "Blobs_vec", index = True)

Blobs_eigenvalue_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/Blobs_Eigenvalue.xlsx", 
                 sheet_name = "Blobs_val", index = True)

'''
#eig_vals_cir, eig_vecs_cir = FF.Ng_spec_decom(sim_mat = sim_cirle, 
#                                                sigma = 0.2, 
#                                                Laplacian = "SYM", I_mat = I)


'''

ARI_Circle_KMeans_df = pd.DataFrame(ari_cir_k)
ARI_Circle_KMeans_df.columns = ["Sigma", "Num. of Clusters", "ARI"]
print(ARI_Circle_KMeans_df)

ARI_Circle_KMeans_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/ARI_KMeans_Circle_large.xlsx", 
                 sheet_name = "Circle", index = True)

opt_sigma_kmeans = ARI_Circle_KMeans_df.iloc[ARI_Circle_KMeans_df.idxmax()["ARI"]]["Sigma"]
print(opt_sigma_kmeans)


ARI_Circle_ALC_df = pd.DataFrame(ari_cir_alc)
ARI_Circle_ALC_df.columns = ["Sigma", "Num. of Clusters", "ARI"]
print(ARI_Circle_ALC_df)


ARI_Circle_ALC_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/ARI_ALC_Circle_large.xlsx", 
                 sheet_name = "Circle", index = True)

opt_sigma_ALC = ARI_Circle_ALC_df.iloc[ARI_Circle_ALC_df.idxmax()["ARI"]]["Sigma"]


'''

'''

eig_vals_blob, eig_vecs_blob = FF.KM_spec_decom(sim_mat = sim_Circle, 
                                                sigma = 24.4545, 
                                                Laplacian = "SYM", I_mat = I)

Circle_eigenvector_df = pd.DataFrame(eig_vecs_blob)
Circle_eigenvalue_df = pd.DataFrame(eig_vals_blob)

Circle_eigenvector_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/Circle_Eigenvector.xlsx", 
                 sheet_name = "Circle_vec", index = True)

Circle_eigenvalue_df.to_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/Circle_Eigenvalue.xlsx", 
                 sheet_name = "Circle_val", index = True)

'''





######################## Plot Hyperparameter  Values ##########################

# Read in the results

Blob_ALC_ARI_df = pd.read_excel("/Users/kirkmeyer/Desktop/UCT/Market State Clustering/ARI_ALC_Blobs_large.xlsx")

Cluster_n1 = Blob_ALC_ARI_df[Blob_ALC_ARI_df["Num. of Clusters"] == 7]

Cluster_n2 = Blob_ALC_ARI_df[Blob_ALC_ARI_df["Num. of Clusters"] == 8]


plt.plot(Cluster_n1["Sigma"], Cluster_n1["ARI"])

plt.plot(Cluster_n2["Sigma"], Cluster_n2["ARI"])



############################# Wishart Distribution ############################


# We need our R matrix: 1/L * A A^T, what is A, is it the Affinity matrix or the data matrix 


# Data Matrix - Circle 
A = blobs
N = blobs.shape[0]
L = blobs.shape[1]
Q = L/N
R = 1/L * np.matmul(A, np.transpose(A))

eig_vals, eig_vecs = la.eigh(R) # SYM eigendecomposition 
        
lambda_max = 1 + 1/Q + 2*np.sqrt(1/Q) 
lambda_min = 1 + 1/Q - 2*np.sqrt(1/Q)

density = Q/(2*np.pi) * (np.sqrt((lambda_max - eig_vals) * (eig_vals - lambda_min)))/eig_vals

indices = np.where(np.isnan(density))
print(indices)

density[indices] = 0
non_zero_idx = np.where(density != 0)


hist_data_1 = np.column_stack((eig_vals[non_zero_idx], density[non_zero_idx]))
hist_data_full = np.column_stack((eig_vals, density))


plt.plot(hist_data_full[:,0], hist_data_full[:,1])
plt.xlabel("$\lambda$")
plt.ylabel("P($\lambda$)")
plt.savefig("Wishart Eigenvalue PDF Plot 1.pdf")

plt.plot(hist_data_1[:,0], hist_data_1[:,1])
plt.xlabel("$\lambda$")
plt.ylabel("P($\lambda$)")
plt.savefig("Wishart Eigenvalue PDF Plot 2.pdf")

# Affinity Matrix 
I = np.eye(len(sim_blobs))
sim_blobs = FF.dist_mat(blobs)

# Create custom colormap
colors = ["darkblue", "white", "darkred"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
sns.heatmap(sim_blobs, vmin = 0, vmax = 600, 
            square=True, cmap = custom_cmap)

plt.savefig("Blob Distance Matrix.pdf")



A_trans = np.exp(-sim_blobs**2 / (2 * 200**2))

sns.heatmap(A_trans, vmin = 0, vmax = 1, 
            square=True, cmap = custom_cmap)

plt.savefig("Blob Similarity Matrix.pdf")





N_sim = A_trans.shape[0]
L_sim = A_trans.shape[1]
Q_sim = L_sim/N_sim
R_sim = 1/L_sim * np.matmul(A_trans, np.transpose(A_trans))

eig_vals_sim, eig_vecs_sim = la.eigh(R_sim)

lambda_max = 1 + 1/Q_sim + 2*np.sqrt(1/Q_sim) 
lambda_min = 1 + 1/Q_sim - 2*np.sqrt(1/Q_sim)

print(Q_sim)
print(lambda_max)
print(lambda_min)

density_sim = Q_sim/(2*np.pi) * (np.sqrt((lambda_max - eig_vals_sim) * (eig_vals_sim - lambda_min)))/eig_vals_sim

hist_data_sim = np.column_stack((eig_vals_sim, density_sim))

#plt.hist(hist_data_sim[:,0], bins=30)

plt.plot(hist_data_sim[:,0], hist_data_sim[:,1])
plt.xlabel("$\lambda$")
plt.ylabel("$p(\lambda)$")

plt.savefig("Blob Wishart Eigenvalue Distribution Plot 1.pdf")



# Data Matrix - Circles 
C = circles
N1 = circles.shape[0]
L1 = circles.shape[1]
Q1 = L1/N1
print(N1,L1,Q1)
R1 = 1/L1 * np.matmul(C, np.transpose(C))


eig_vals_circle, eig_vec_circle = la.eigh(R1) 

lambda_max_1 = 1 + 1/Q1 + 2*np.sqrt(1/Q1) 
lambda_min_1 = 1 + 1/Q1 - 2*np.sqrt(1/Q1)


density_1 = Q1/(2*np.pi) * (np.sqrt((lambda_max_1 - eig_vals_circle) * (eig_vals_circle - lambda_min_1)))/eig_vals_circle




def KM_Wishart_dist(data):
    
    N = data.shape[0]
    L = data.shape[1]
    Q = L/N # generally L is longer than N 

    A = data
    
    R = 1/L * np.matmul(A, np.transpose(A))
    
    eigenvalues, eigenvector = la.eigh(R)
    
    
    lambda_max = 1 + 1/Q + 2*np.sqrt(1/Q) 
    lambda_min = 1 + 1/Q - 2*np.sqrt(1/Q)
    
    
    density = Q/(2*np.pi) * (np.sqrt((lambda_max - eigenvalues) * (eigenvalues - lambda_min)))/eigenvalues
    
    return density, lambda_max, lambda_min, Q


# Blob raw data 

    
blob_lambda_density, blob_l_max, blob_l_min, blob_Q  = KM_Wishart_dist(data = blobs)

# Blob Affinity Matrix

blob_A_lambda_density, blob_A_l_max, blob_A_l_min, blob_A_Q  = KM_Wishart_dist(data = blobs)








######################### Blobs Eigenvector Resampling ########################


dim1 = len(sim_blobs)
print(dim1)
sample_perc = np.array([1, 0.9, 0.7, 0.5, 0.3, 0.1])

indices = np.arange(0,dim1)
print(indices)


# Create custom colormap
colors = ["darkblue", "white", "darkred"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    
heat_map_dict = {}


for p in sample_perc:
    

    sampled_vectors_1 = np.random.choice(indices, 
                                         size = int(dim1 * p), 
                                         replace = False)
    
    sampled_vectors_2 = np.random.choice(indices, 
                                       size = int(dim1 * p), 
                                       replace = False)
    
    sample_data_1 = blobs[sampled_vectors_1, :]
    sample_data_2 = blobs[sampled_vectors_2, :]
    
    sim_blobs_1 = FF.dist_mat(sample_data_1)
    sim_blobs_2 = FF.dist_mat(sample_data_2)
    
    I_1 = np.eye(len(sim_blobs_1))
        
    eig_vals_blob_1, eig_vecs_blob_1 = FF.KM_spec_decom(sim_mat = sim_blobs_1, 
                                                    sigma = 44.21, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    eig_vals_blob_2, eig_vecs_blob_2 = FF.KM_spec_decom(sim_mat = sim_blobs_2, 
                                                    sigma = 44.21, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    lead_vecs_dim = 15
    
    dot_mat = np.zeros((lead_vecs_dim, lead_vecs_dim))
    
    
    
    for i in range(len(dot_mat)):
        
        vect_1 = eig_vecs_blob_1[:,i] 
        
        for j in range(len(dot_mat)):
            
            vect_2 = eig_vecs_blob_2[:,j] 
            
            dot_mat[i,j] = np.dot(vect_1, vect_2)
    
    heat_map_dict[f"HeatMat{p}"] = dot_mat
    
    sns.heatmap(dot_mat, vmin = np.min(dot_mat), vmax = np.max(dot_mat), square=True, cmap = custom_cmap)
    percent_done = str(p*100) + "% Done"
    print(percent_done)
    #plt.savefig("Eigenstability Test 85%.pdf")
    
    file_name = "Eigenstability_Test_" + str(p*100) + "%.pdf"
    
    plt.savefig(file_name)
    
    plt.close()




'''

# Finding max and min across all groups

min_heatmap = np.min(np.array([np.min(heat_map_dict[f"HeatMat1.0"]), np.min(heat_map_dict[f"HeatMat0.9"]),
                       np.min(heat_map_dict[f"HeatMat0.7"]), np.min(heat_map_dict[f"HeatMat0.5"]),
                       np.min(heat_map_dict[f"HeatMat0.3"]), np.min(heat_map_dict[f"HeatMat0.1"])]))


max_heatmap = np.max(np.array([np.max(heat_map_dict[f"HeatMat1.0"]), np.max(heat_map_dict[f"HeatMat0.9"]),
                       np.max(heat_map_dict[f"HeatMat0.7"]), np.max(heat_map_dict[f"HeatMat0.5"]),
                       np.max(heat_map_dict[f"HeatMat0.3"]), np.max(heat_map_dict[f"HeatMat0.1"])]))

'''

'''
fig, axs = plt.subplots(2, 3, figsize=(10, 8))

heatmap_0 = sns.heatmap(heat_map_dict[f"HeatMat1.0"], vmin = min_heatmap, 
                      vmax = max_heatmap, square=True, cmap = custom_cmap, ax = axs[0,0])
axs[0,0].set_title("Full Data Set")

heatmap_1 = sns.heatmap(heat_map_dict[f"HeatMat0.9"], vmin = min_heatmap, 
                      vmax = max_heatmap, square=True, cmap = custom_cmap, ax = axs[0,1])
axs[0,1].set_title("90% Resample")

heatmap_2 = sns.heatmap(heat_map_dict[f"HeatMat0.7"], vmin = min_heatmap, 
                      vmax = max_heatmap, square=True, cmap = custom_cmap, ax = axs[0,2])
axs[0,2].set_title("70% Resample")

heatmap_3 = sns.heatmap(heat_map_dict[f"HeatMat0.5"], vmin = min_heatmap, 
                      vmax = max_heatmap, square=True, cmap = custom_cmap, ax = axs[1,0])
axs[1,0].set_title("50% Resample")


heatmap_4 = sns.heatmap(heat_map_dict[f"HeatMat0.3"], vmin = min_heatmap, 
                      vmax = max_heatmap, square=True, cmap = custom_cmap ,ax = axs[1,1])
axs[1,1].set_title("30% Resample")

heatmap_5= sns.heatmap(heat_map_dict[f"HeatMat0.1"], vmin = min_heatmap, 
                      vmax = max_heatmap, square=True, cmap = custom_cmap, ax = axs[1,2])
axs[1,2].set_title("10% Resample")
'''

fig, axs = plt.subplots(2, 3, figsize=(10, 8))

heatmap_0 = sns.heatmap(heat_map_dict[f"HeatMat1.0"], vmin = np.min(heat_map_dict[f"HeatMat1.0"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat1.0"]), square=True, cmap = "coolwarm", ax = axs[0,0])
axs[0,0].set_title("Full Data Set")

heatmap_1 = sns.heatmap(heat_map_dict[f"HeatMat0.9"], vmin = np.min(heat_map_dict[f"HeatMat0.9"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.9"]), square=True, cmap = "coolwarm", ax = axs[0,1])
axs[0,1].set_title("90% Resample")

heatmap_2 = sns.heatmap(heat_map_dict[f"HeatMat0.7"], vmin = np.min(heat_map_dict[f"HeatMat0.7"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.7"]), square=True, cmap = "coolwarm", ax = axs[0,2])
axs[0,2].set_title("70% Resample")

heatmap_3 = sns.heatmap(heat_map_dict[f"HeatMat0.5"], vmin = np.min(heat_map_dict[f"HeatMat0.5"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.5"]), square=True, cmap = "coolwarm", ax = axs[1,0])
axs[1,0].set_title("50% Resample")


heatmap_4 = sns.heatmap(heat_map_dict[f"HeatMat0.3"], vmin = np.min(heat_map_dict[f"HeatMat0.3"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.3"]), square=True, cmap = "coolwarm" ,ax = axs[1,1])
axs[1,1].set_title("30% Resample")

heatmap_5= sns.heatmap(heat_map_dict[f"HeatMat0.1"], vmin = np.min(heat_map_dict[f"HeatMat0.1"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.1"]), square=True, cmap = "coolwarm", ax = axs[1,2])
axs[1,2].set_title("10% Resample")



colorbar = heatmap_0.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting

colorbar = heatmap_1.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    
colorbar = heatmap_2.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    

colorbar = heatmap_3.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    
colorbar = heatmap_4.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    
colorbar = heatmap_5.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    

plt.savefig("Overlap Matrices.pdf")

plt.close()


################## Full Rank Blobs Eigenvector Resampling ####################

full_heat_map_dict = {}
dim1 = len(sim_blobs)

for p in sample_perc:
    

    sampled_vectors_1 = np.random.choice(indices, 
                                         size = int(dim1 * p), 
                                         replace = False)
    
    sampled_vectors_2 = np.random.choice(indices, 
                                       size = int(dim1 * p), 
                                       replace = False)
    
    sample_data_1 = blobs[sampled_vectors_1, :]
    sample_data_2 = blobs[sampled_vectors_2, :]
    
    sim_blobs_1 = FF.dist_mat(sample_data_1)
    sim_blobs_2 = FF.dist_mat(sample_data_2)
    
    I_1 = np.eye(len(sim_blobs_1))
        
    eig_vals_blob_1, eig_vecs_blob_1 = FF.KM_spec_decom(sim_mat = sim_blobs_1, 
                                                    sigma = 44.21, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    eig_vals_blob_2, eig_vecs_blob_2 = FF.KM_spec_decom(sim_mat = sim_blobs_2, 
                                                    sigma = 44.21, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    lead_vecs_dim = len(sim_blobs_1)
    
    dot_mat = np.zeros((lead_vecs_dim, lead_vecs_dim))
    
    
    
    for i in range(len(dot_mat)):
        
        vect_1 = eig_vecs_blob_1[:,i] 
        
        for j in range(len(dot_mat)):
            
            vect_2 = eig_vecs_blob_2[:,j] 
            
            dot_mat[i,j] = np.dot(vect_1, vect_2)
    
    full_heat_map_dict[f"HeatMat{p}"] = dot_mat
    
    sns.heatmap(dot_mat, vmin = np.min(dot_mat), vmax = np.max(dot_mat), square=True, cmap = custom_cmap)
    percent_done = str(p*100) + "% Done"
    print(percent_done)
    #plt.savefig("Eigenstability Test 85%.pdf")
    
    file_name = "Full_Eigenstability_Test_" + str(p*100) + "%.pdf"
    
    plt.savefig(file_name)
    
    plt.close()

###############################################################################



######################### Circle Eigenvector Resampling #######################


dim1 = len(sim_circle)
print(dim1)
sample_perc = np.array([1, 0.9, 0.7, 0.5, 0.3, 0.1])

indices = np.arange(0,dim1)
print(indices)


# Create custom colormap
colors = ["darkblue", "white", "darkred"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

for p in sample_perc:
    

    sampled_vectors_1 = np.random.choice(indices, 
                                         size = int(dim1 * p), 
                                         replace = False)
    
    sampled_vectors_2 = np.random.choice(indices, 
                                       size = int(dim1 * p), 
                                       replace = False)
    
    sample_data_1 = circles[sampled_vectors_1, :]
    sample_data_2 = circles[sampled_vectors_2, :]
    
    sim_circle_1 = FF.dist_mat(sample_data_1)
    sim_circle_2 = FF.dist_mat(sample_data_2)
    
    I_1 = np.eye(len(sim_circle_1))
    
    eig_vals_blob_1, eig_vecs_blob_1 = FF.KM_spec_decom(sim_mat = sim_circle_1, 
                                                    sigma = 0.25, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    eig_vals_blob_2, eig_vecs_blob_2 = FF.KM_spec_decom(sim_mat = sim_circle_2, 
                                                    sigma = 0.25, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    lead_vecs_dim = 15
    
    dot_mat = np.zeros((lead_vecs_dim, lead_vecs_dim))
    
    
    
    for i in range(len(dot_mat)):
        
        vect_1 = eig_vecs_blob_1[:,i] 
        
        for j in range(len(dot_mat)):
            
            vect_2 = eig_vecs_blob_2[:,j] 
            
            dot_mat[i,j] = np.dot(vect_1, vect_2)
            
            
            
    sns.heatmap(dot_mat, vmin = np.min(dot_mat), vmax = np.max(dot_mat), square=True, cmap = custom_cmap)
    percent_done = str(p*100) + "% Done"
    print(percent_done)
    #plt.savefig("Eigenstability Test 85%.pdf")
    
    file_name = "Circle Eigenstability_Test_" + str(p*100) + "%.pdf"
    
    plt.savefig(file_name)
    
    plt.close()
 
###############################################################################
                          ## Sub Plot Circle ##
###############################################################################



fig, axs = plt.subplots(2, 3, figsize=(10, 8))

heatmap_0 = sns.heatmap(heat_map_dict[f"HeatMat1.0"], vmin = np.min(heat_map_dict[f"HeatMat1.0"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat1.0"]), square=True, cmap = "coolwarm", ax = axs[0,0])
axs[0,0].set_title("Full Data Set")

heatmap_1 = sns.heatmap(heat_map_dict[f"HeatMat0.9"], vmin = np.min(heat_map_dict[f"HeatMat0.9"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.9"]), square=True, cmap = "coolwarm", ax = axs[0,1])
axs[0,1].set_title("90% Resample")

heatmap_2 = sns.heatmap(heat_map_dict[f"HeatMat0.7"], vmin = np.min(heat_map_dict[f"HeatMat0.7"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.7"]), square=True, cmap = "coolwarm", ax = axs[0,2])
axs[0,2].set_title("70% Resample")

heatmap_3 = sns.heatmap(heat_map_dict[f"HeatMat0.5"], vmin = np.min(heat_map_dict[f"HeatMat0.5"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.5"]), square=True, cmap = "coolwarm", ax = axs[1,0])
axs[1,0].set_title("50% Resample")


heatmap_4 = sns.heatmap(heat_map_dict[f"HeatMat0.3"], vmin = np.min(heat_map_dict[f"HeatMat0.3"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.3"]), square=True, cmap = "coolwarm" ,ax = axs[1,1])
axs[1,1].set_title("30% Resample")

heatmap_5= sns.heatmap(heat_map_dict[f"HeatMat0.1"], vmin = np.min(heat_map_dict[f"HeatMat0.1"]), 
                      vmax = np.max(heat_map_dict[f"HeatMat0.1"]), square=True, cmap = "coolwarm", ax = axs[1,2])
axs[1,2].set_title("10% Resample")



colorbar = heatmap_0.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting

colorbar = heatmap_1.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    
colorbar = heatmap_2.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    

colorbar = heatmap_3.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    
colorbar = heatmap_4.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    
colorbar = heatmap_5.collections[0].colorbar  # Access color bar
colorbar.formatter = ticker.ScalarFormatter(useMathText=True)
colorbar.formatter.set_scientific(True)
colorbar.formatter.set_powerlimits((0, 0))  # Enforces scientific notation
colorbar.update_ticks()  # Update the color bar with the new formatting
    

plt.savefig("Overlap Matrices.pdf")

plt.close()


################## Full Rank Blobs Eigenvector Resampling ####################

full_heat_map_dict = {}
dim1 = len(sim_blobs)

for p in sample_perc:
    

    sampled_vectors_1 = np.random.choice(indices, 
                                         size = int(dim1 * p), 
                                         replace = False)
    
    sampled_vectors_2 = np.random.choice(indices, 
                                       size = int(dim1 * p), 
                                       replace = False)
    
    sample_data_1 = blobs[sampled_vectors_1, :]
    sample_data_2 = blobs[sampled_vectors_2, :]
    
    sim_blobs_1 = FF.dist_mat(sample_data_1)
    sim_blobs_2 = FF.dist_mat(sample_data_2)
    
    I_1 = np.eye(len(sim_blobs_1))
        
    eig_vals_blob_1, eig_vecs_blob_1 = FF.KM_spec_decom(sim_mat = sim_blobs_1, 
                                                    sigma = 44.21, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    eig_vals_blob_2, eig_vecs_blob_2 = FF.KM_spec_decom(sim_mat = sim_blobs_2, 
                                                    sigma = 44.21, 
                                                    Laplacian = "SYM", I_mat = I_1)
    
    
    lead_vecs_dim = len(sim_blobs_1)
    
    dot_mat = np.zeros((lead_vecs_dim, lead_vecs_dim))
    
    
    
    for i in range(len(dot_mat)):
        
        vect_1 = eig_vecs_blob_1[:,i] 
        
        for j in range(len(dot_mat)):
            
            vect_2 = eig_vecs_blob_2[:,j] 
            
            dot_mat[i,j] = np.dot(vect_1, vect_2)
    
    full_heat_map_dict[f"HeatMat{p}"] = dot_mat
    
    sns.heatmap(dot_mat, vmin = np.min(dot_mat), vmax = np.max(dot_mat), square=True, cmap = custom_cmap)
    percent_done = str(p*100) + "% Done"
    print(percent_done)
    #plt.savefig("Eigenstability Test 85%.pdf")
    
    file_name = "Full_Eigenstability_Test_" + str(p*100) + "%.pdf"
    
    plt.savefig(file_name)
    
    plt.close()

###############################################################################















###############################################################################













    