#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:57:06 2024

@author: kirkmeyer

This file houses all the functions required to perform the necessary experiments
within my MSc. dissertation. 

"""

# Reading in external packages required in 

import numpy as np
from numba import jit
import itertools
from sklearn import metrics
from scipy.spatial.distance import cdist
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



"""
      ############### Cluster Likelihood Function ###############

"""

''' the cluster function:
    compute the likelihood which occurs when two objects are clustered or
    the object own likelihood. By design if a cluster only containts one object
    its likelihood will be 0'''
@jit(nopython=True)
def clus_lc(gij,gii,gjj,method,ns=2):
    ''' variables description:
        ns is the size of the cluster.
        cs is the intracluster correlation.
        gij, gii, and gjj respectively are
        correlations relating to interactions between objects i and j, and
        their respective self-correlations. self-correlations (i.e. gii, gjj)
        are 1 for individual objects and >1 for clusters'''
        
    if ns==1:
        return 0
    ''' intracluster correlation'''   
    cs = 2*gij+gii+gjj-1e-3
    ''' relatively low cs means noisy and suboptimal clusters.
    The coupling parameter gs (see paper) isn't not defined'''
    if cs<=ns:
        return 0
    
    if method == "GM":
        return 0.5 * ( np.log(ns/cs) +  (ns - 1)*np.log( (ns**2 - ns) / ( ns**2 - cs) )  )

    if method == "KM":
        return  ns - ns/cs

''' aspc only requires a correlation matrix as input:
    here we convert the correlation to a dictionary for convenience. adding new
    entries in a dict() is much faster than editing a numpy matrix'''
    

"""

 ############### Agglomerative Likelihood Clustering Algorithm ###############

"""

''' aspc only requires a correlation matrix as input:
    here we convert the correlation to a dictionary for convenience. adding new
    entries in a dict() is much faster than editing a numpy matrix'''
    
    
    
    
def alc(G, method, cn= None):
    
    # cn is number of clusters
    N = len(G) # number of rows  (or columns since corr mat is square)
    gdic = dict(enumerate(dict(enumerate(row)) for row in G)) 
    
    ''' This line of code is creating a dictionary of dictionaries. 
    Each inner dictionary corresponds to a row in G, where each element 
    in the row is indexed by its position. The outer dictionary indexes these 
    inner dictionaries by their row position in G. If G is a list of lists (like a matrix), 
    the code essentially maps each row to a dictionary where the keys are the column 
    indices, and then maps each row (as a dictionary) to its row index.'''
    
    clike = { i : 0 for i in range(N)} # dictionary where key value is index of data point and value is set to zero. 
    del G
    
    ''' tracker is dictionary which stores the objects member of the same clusters.
        the data is stored as strings: i.e. cluster 1234 contains objects 210 & 890
        which results in tracker['1234'] == '210_890' '''
    tracker = { i:[i] for i in range(N) }
    
    # ''' the cluster size ns is stored in the ns_ array'''
    # ns_ = [1]*N
    
    ''' Create a list of object indices 'other_keys': at every iteration one object
     is clustered and removed from the list. It is also removed if no suitable
     optimal destination is found.'''
    other_keys = list(range(N)) # keeps track of the objects that at each iteration have not yet been considered 
    
    ''' the operation stops once there is only one object left to cluster as we
    need two objects at the very least.'''
    while len(tracker) != cn:
        
        ''' a random initialization:
            pick a object 'node' at random to start clustering,
            this might have a consequence on the final result depending on the data.
            then loop through the other objects using 'nbor' and costs to store
            the likelihood resulting from clustering 'node' to the objects in
            'nbor'.
            indices: stores the indices which are combinations of node and others.
            costs: stores the cost which compute the difference between the likelihood
            of the resulting cluster and the sum of the two individual objects forming
            the result cluster.
            '''
        ''' the routine uses other_keys and removes elements everytime they are clustered
        or can't be clustered anymore. If a cluster number is not provided the routine
        stops there. If one is then it continues by looking at the elements in the
        optimal cluster solution (tracker) and continues merging until the preset
        number of clusters is met'''
        if len(other_keys)>1:
            node = np.random.choice(other_keys) # choosing the random starting point of thosr not yet selected
        
        else: # meaning we have gone through all the points and everything is in some sort of cluster (singleton or not),
        # meaning we are now looking to see optimal merges.
            if len(tracker) != 1:
                # This line of code randomly selects one key (which is a cluster index) from the tracker dictionary.
                node = np.random.choice(list(tracker.keys())) 
            else:
                cn = 1 # if both conditions can't be met, then there is only cluster
                continue
        nbor = list(tracker.keys())
        nbor.remove(node) # remove starting node from neighbourhood nodes it can be clustered with.
        costs = np.zeros(len(nbor)) #initialise cost vector for interation
        indices = [(node,key) for key in nbor] #creating an exhuastive list of testing the node and every other data point
        node_lc = clus_lc(0,gdic[node][node],0,method,ns = len(tracker[node])) 
        k=0
        
        ''' The loop goes through each tuple in the list indices, and in each 
        iteration, it assigns the first element of the tuple to i and the 
        second element to j. You can then use i and j to perform operations within the loop.'''
        
        for i,j in indices:
            costs[k] = clus_lc(gdic[i][j],gdic[i][i],gdic[j][j],method,ns=len(tracker[i])+len(tracker[j])) - (node_lc+clus_lc(0,gdic[j][j],0,method,ns = len(tracker[j])))
            k+=1
            
        ''' find the optimal cost which will be the object clustered with node'''
        next_merge = np.argmax(costs) # taking the maximum change in likelihood 
        
        
        ''' stopping conditions '''
        if costs[next_merge]<=0:
            if len(other_keys)>1:
                ''' if no cost is positive then this node cannot be clustered further
                and must be removed from the list'''
                other_keys.remove(node)
                continue
            elif not cn:
                ''' if no cluster number is provided then the routine has completed
                and tracker is the final solution'''
                cn = len(tracker)
                continue
            elif cn:
                ''' if a cluster number is provided the routine continues and keeps
                merging'''
                pass
        ''' on the other hand, the largest positive cost is the designated
        object 'label_b' clustered to node which here is stored as 'label_a'.
        new clusters 'new_label' take values superior to N.
        tracker, as previously explained, stores joined strings of the clusters
        contents'''
        
        label_a = node
        label_b = indices[next_merge][1]
        new_label = list(tracker.keys())[-1]+1
        clike[new_label] = clus_lc(gdic[label_a][label_b],gdic[label_a][label_a],gdic[label_b][label_b],method,ns=len(tracker[label_a])+len(tracker[label_b])) 
        del clike[label_a]
        del clike[label_b]
        ''' removes individual elements which were merged and update others with the new cluster.
        only do it when a positive cost is found.'''
        if costs[next_merge]>0:
            other_keys = list(tracker.keys())
            other_keys.remove(label_a)
            other_keys.remove(label_b)
            other_keys.append(new_label)
    
        
        ''' Once a cluster is formed, the correlation matrix gdic and tracker need to
        be updated with the new cluster and the cluster size must be updated with ns_'''
        nbor.remove(label_b)
        tracker[new_label]=tracker[label_a] + tracker[label_b]
        gdic[new_label]={} # new key + value pair added. Initialised by empty value
        gdic[new_label][new_label] = 2*gdic[label_a][label_b] + gdic[label_a][label_a] + gdic[label_b][label_b] 
        # new self correlation added into value for new label key
    
        for key in nbor:
            # adding cross correlations (off diagonal)
            gdic[new_label][key] = gdic[label_a][key] + gdic[label_b][key]
            gdic[key][new_label] = gdic[label_a][key] + gdic[label_b][key]
    
        del tracker[label_a]
        del tracker[label_b]
    
    
    ''' create the final clustering array:
        tracker contains the cluster memberships but as a dictionary
        we create a numpy array where stocked are labeled with the same number
        if they belong to the same cluster, and 0 if unclustered'''
        
    solution = np.zeros(N,dtype=int)
    k=1
    for cluster in tracker.keys():
        cluster_members = tracker[cluster]
        solution[cluster_members] = k
        k+=1
    return solution

"""

############### Agglomerative Likelihood Clustering resamplying ###############



sizes = range(1000,3500,500)
T = range(10,35,5)
aggdata = { i : {} for i in sizes}
clusters_number = 10
gs = 1 
model = 'normal'
 
for N, t in zip(sizes, T):    

    data, key = onefactor(N, clusters_number,t,coupling_parameter = gs, model=model)
    print('noise signal ratio', t/N)    
    
    ''' ALC '''
    sample_size = int(1/(0.1/t))
    
    spin = { i : {j : 0 for j in range(i+1,N)} for i in range(N-1)}
    freq = { i : {j : 0 for j in range(i+1,N)} for i in range(N-1)}
    ari=0
    tot_iter = 0
    while ari <.9 and tot_iter<=2000:
        
        for iterations in range(100):
            idx = np.random.choice(range(N),size=sample_size,replace=False)
            for i, j in list(itertools.combinations(idx,2)):
                mi, ma = min([i,j]),max([i,j])
                freq[mi][ma]+=1
            temp = data[idx]
            aspc_solution = alc(temp)
            labels = np.unique(aspc_solution)
            indices = [ idx[aspc_solution == label] for label in labels]
            for idx_ in indices:
                for i,j in list(itertools.combinations(idx_,2)):
                    mi, ma = min([i,j]),max([i,j])
                    spin[mi][ma]+=1
        tot_iter+=100
        pdf = np.zeros((N,N))
        for i,j in list(itertools.combinations(range(N),2)): 
            try: pdf[i,j] = spin[i][j]/freq[i][j]
            except:
                continue
        pdf_ = (pdf + pdf.T)
        
        thres=.5
        tracker = { i:i for i in range(N)}
        
        for i in range(N):
            idx = np.arange(N,dtype=int)[pdf_[i]>=thres]
            for j in idx:
                tracker[j] = tracker[i]
        t_solution = list(tracker.values())
        solution =np.ones(N)*-1
        k=0
        for i in np.unique(t_solution):
            solution[t_solution==i]=k
            k+=1
        ari = metrics.adjusted_rand_score(key,solution)
        aggdata[N][tot_iter] = ari
        print(ari)

# np.save('surrogate_ari05.npy', aggdata)
# data = np.load('surrogate_ari.npy',allow_pickle=True).item()
# T = list(T)
# k=0
# plt.figure()
# for key in data.keys():
#     x = list(data[key].keys())
#     y = list(data[key].values())
#     plt.scatter(x,y,label = r'N=%s n=%s t=%s' % (key,int(key/10),T[k]))
#     plt.plot(x,y)
#     plt.xlabel('iterations')
#     plt.ylabel('ari')
#     # plt.legend()
#     k+=1
# plt''.tight_layout()

"""

"""
         ############### K-Means Clustering Algorithm ###############

"""

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


"""
         ############### Distance Matrix ###############

"""

def dist_mat(X):
    S = cdist(X, X, 'euclidean') # finds euclidean distance
    return S
   
"""
         ############### Renormalise ###############

""" 


def renorm(x):
    return x / np.sqrt(np.sum(x**2))



"""
        ################### Spectral Decomposition  ###################

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
        #L_sym   = D_2 @ A @ D_2 
        eig_vals, eig_vecs_sym = la.eigh(L_sym) # SYM eigendecomposition 
        
        eig_vecs = renorm(eig_vecs_sym)
        
        
        
    return eig_vals, eig_vecs



"""
        ################### Ng Spectral Decomposition  ###################

"""

# Setting up the Laplaciansx

def Ng_spec_decom(sim_mat, sigma, Laplacian, I_mat):
    
    
    A = np.exp(-sim_mat**2 / (2 * sigma**2)) # Gaussian Affinity matrix


    if Laplacian == "RW":
        
        D_1     = np.diag(1/np.sum(A,axis = 1)) #D^{-1}
        L_rw    = I_mat  - D_1 @ A
        eig_vals, eig_vecs  = la.eigh(L_rw) # RW eigendecomposition 



        
    
    
    if Laplacian == "SYM":
        
        D_2     = np.diag(1 / np.sqrt(np.sum(A, axis=1))) #D^{-1/2}
        L_sym   = D_2 @ A @ D_2 
        eig_vals, eig_vecs_sym = la.eigh(L_sym) # SYM eigendecomposition 
        
        eig_vecs = renorm(eig_vecs_sym)
        
        
        
    return eig_vals, eig_vecs








"""
        ################### Circle Data Generation  ###################

"""

def KM_makecircle(n_points, n_circles):
    
    label = np.random.choice(range(1,n_circles+1), n_points, replace=True)
    r = 2*label + np.random.normal(0, 0.1, n_points)
    X = np.random.uniform(-1, 1, n_points)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
        
    X = np.column_stack((x, y))
    
    cir_plot = plt.scatter(x, y, c = label,  s = 50, cmap='viridis')
    
    return cir_plot, label, X

"""
        ################### Blob Data Generation  ###################

"""


def KM_makeblobs(N, num_clusters):
    blobs,labels = make_blobs(n_samples=N,
                             n_features=5000,
                             shuffle=False,
                             centers=num_clusters,
                             random_state=0)
    
    blob_plot = plt.scatter(blobs[:, 0], blobs[:, 1], c=labels, s=50, cmap='viridis')
    
    return blob_plot, blobs, labels

"""
        ################### Hyperparameter Tuning ###################

"""

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





