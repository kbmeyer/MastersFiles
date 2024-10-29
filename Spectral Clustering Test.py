_#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:22:46 2024

Here we look to create some sort of "run file" for the concentric circle data 
test case. The file will consider the follow: 
    
    A.) Clustering Algos: 
        
        1.) EP's KMeans function 
        2.) LY's ALC function 
    
    B.) Spectral Clustering Scheme:
        
        1.) A generalised function where we perform vanilla SC(no looking into 
            different graph Lapclians or similarity matrices)
    
    C.) Searching for optimal sigma value

                                                               
@author: KM + TG
"""

""" Importing required packages """

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from numba import jit
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D


""" Clustering Algos """

''' the likelihood cluster function:
    compute the likelihood which occurs when two objects are clustered or
    the object own likelihood. By design if a cluster only containts one object
    its likelihood will be 0. Also, can specify which version of ALC is used '''

def clus_lc(gij,gii,gjj,method,ns=3):
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
        return 0.5*( np.log(ns/cs) +  (ns - 1)*np.log( (ns**2 - ns) / ( ns**2 - cs) )  )

    if method == "KM":
        return  ns - ns/cs

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


""" Simulating the test data """

# Simulation Function 
def sim(N=500):
    r = np.random.choice([2, 4, 6], N, replace=True) + np.random.normal(0, 0.1, N)
    X = np.random.uniform(-1, 1, N)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
    return x, y, r

# Creating the data 
dat_X1, dat_X2, dat_r = sim()

# Data in X-space
X = np.column_stack((dat_X1, dat_X2))

""" Spectral Clustering function """

def SpecCluster(Data, Clustering_Method, Sigma, Eigen_CutOff, Figures):
    
    if Clustering_Method == "ALC_GM":
        
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
                return 0.5*( np.log(ns/cs) +  (ns - 1)*np.log( (ns**2 - ns) / ( ns**2 - cs) )  )

            if method == "KM":
                return  ns - ns/cs
            
            
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
                ''' removes merged elements and update others with the new cluster.
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

        
    
    

























