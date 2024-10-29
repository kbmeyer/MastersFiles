#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:14:01 2024

@author: kirkmeyer
"""


from Function_File import alc, clus_lc, k_means, dist_mat, renorm, KM_spec_decom, KM_makecircle, KM_makeblobs
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score

random.seed(12)

circle_plot, label, data = KM_makecircle(n_points = 1000, n_circles = 3)

sim_mat = dist_mat(data)
I = np.eye(len(sim_mat))


def KM_cluster_tuning(data, true_labels, n_sigmas, start_sigma, end_sigma, 
                      Laplacian, range_n_clusters, method, plot):
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import random
    from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
    
    true_labels = true_labels
    sim_matrix = dist_mat(data)
    I = np.eye(len(sim_matrix))
    sigmas = np.linspace(start_sigma, end_sigma, n_sigmas)
    
    SIL_store = []            # Create a DataFrame to store values
    ARI_store = []            # Create a DataFrame to store values
    
    for sigma in enumerate(sigmas):
        
        eig_vals, eig_vecs = KM_spec_decom(sim_mat = sim_matrix, 
                                           sigma = sigma[1], Laplacian = Laplacian, 
                                           I_mat = I)
    
        for n_clusters in range_n_clusters:
            
            XX = eig_vecs[:,: n_clusters]
            
            if method == "LY ALC":
                
                corr_mat = np.corrcoef(XX)
                cluster_labels = alc(corr_mat, method = "GM")
                
            
            if method == "ET K-Means":
                
                initial_idx =np.array([random.randint(0,len(data)-1) for _ in range(n_clusters)])
                res_clust_Mus, res_clust_labels, res_clust_dists, res_clust_K, res_clust_error = k_means(XX, initial_idx, 10000)
                cluster_labels = res_clust_labels
            
            
            
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
                        
            silhouette_avg = silhouette_score(XX, cluster_labels)
            ARI = adjusted_rand_score(true_labels, cluster_labels)
            
            SIL_store.append([sigma[1], n_clusters, silhouette_avg])
            ARI_store.append([sigma[1], n_clusters, ARI])
        
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
            
            if plot == "True":

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
            
                # Compute the silhouette scores for each sample, needed for the plots
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
                
                
    return SIL_store, ARI_store



sil_ALC, ari_ALC = KM_cluster_tuning(data = data, true_labels = label, n_sigmas = 20, 
                             start_sigma = 0.15, end_sigma = 1, 
                             Laplacian = "SYM", range_n_clusters = [3,4,5], 
                             method = "LY ALC",  
                             plot = "False")


eig_vals, eig_vecs = KM_spec_decom(sim_mat = sim_mat, 
                                   sigma = 0.15, Laplacian = "SYM", 
                                   I_mat = I)

V = eig_vecs[:,:9]

corr_mat = np.corrcoef(V)
cluster_labels = alc(corr_mat, method = "GM")

plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, s=50, cmap='viridis')

silhouette_avg = silhouette_score(V, cluster_labels)
print(silhouette_avg)
ARI = adjusted_rand_score(label, cluster_labels)
print(ARI)




sil_kmeans, ari_kmeans  = KM_cluster_tuning(data = data, true_labels = label, n_sigmas = 10, 
                             start_sigma = 0.05, end_sigma = 1, 
                             Laplacian = "SYM", range_n_clusters = [7,8,9], 
                             method = "ET K-Means",  
                             plot = "False")




plt.show()

