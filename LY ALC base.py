'''Project: “Agglomerative Likelihood Clustering”
Author : Lionel Yelibi, 2019, University of Cape Town.
Copyright SPC, 2019, 2020, 2021
Potts Model Clustering.
Agglomerative Likelihood Clustering
See pre-print: https://arxiv.org/abs/1908.00951
GNU GPL
This file is part of ALC
ALC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
ALC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.'''

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from numba import jit
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.linalg as la 
from sklearn import metrics
import seaborn as sns


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



''' The line of code below generates a synthetic dataset with N samples, 
each having 5000 features, grouped into 10 distinct clusters, without 
shuffling the data. The resulting dataset is stored in the variable data, 
while the cluster labels are generated but ignored (hence the _). '''


''' ######################## LY's Blob Data Test ##########################'''


N = 500
data,labels = make_blobs(n_samples=N,n_features=5000,shuffle=False,centers=10,random_state=0)

M = 100 
sigmas = np.linspace(0.05, 10, M)
cluster_quality = np.zeros(M)
I = np.eye(500)

def dist_mat(X):
    S = cdist(X, X, 'euclidean') # finds euclidean distance
    return S

res_S = dist_mat(data) 

def renorm(x):
    return x / np.sqrt(np.sum(x**2))


for i,sig in enumerate(sigmas):
    
    A = np.exp(-res_S**2 / (2 * sig**2)) # Gaussian Affinity matrix
    D = np.diag(1 / np.sqrt(np.sum(A, axis=1))) # finding D^{-1/2}, axis=1 is summing across the rows. This is the degree matrix p.2 from Tut of SC
    #L = D @ A @ D # normalised symmetric laplacian (Ng, Jordan and Weiss)
    L_sym = I - D @ A @ D # normalised symmetric laplacian (Tutorial on spectral clustering)
    # L_rw  = 

    
    res2_vals, res2_vecs = la.eigh(L_sym) # eigendecomposition 
    #idx = res2_vals.argsort()[::-1] # finding indices for reordering due to SciPy eigh function output 
    #sorted_eigenvalues = res2_vals[idx] 
    #sorted_eigenvectors = res2_vecs[:, idx]
    Vnorm = np.apply_along_axis(renorm, 1, res2_vecs)

    
    ALC_spec_GM = alc(np.corrcoef(Vnorm),"GM")
    cluster_quality[i] = metrics.adjusted_rand_score(labels, ALC_spec_GM)

max_ARI = np.max(cluster_quality)
print(max_ARI)
sigma = sigmas[np.argmax(cluster_quality)] # need to test a range of these for maximise/minimise some objective function
print(sigma)

   



corr_blobs = np.corrcoef(data)
sol_GM  = alc(np.corrcoef(data), "GM")
sol_KM  = alc(np.corrcoef(data), "KM")

plt.figure(figsize=(20,18))  # Optional: Set the size of the plot
sns.heatmap(corr_blobs, annot=False, cmap='viridis', fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
plt.show()

plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')

plt.scatter(data[:, 0], data[:, 1], c=sol_GM, s=50, cmap='viridis')
print(np.unique(sol_GM))


plt.scatter(data[:, 0], data[:, 1], c=sol_KM[0], s=50, cmap='viridis')
print(np.unique(sol_KM[0]))

sigma = 1


I = np.eye(len(res_S))
A = np.exp(-res_S**2 / (2 * sigma**2)) # Gaussian Affinity matrix
D = np.diag(1 / np.sqrt(np.sum(A, axis=1))) # finding D^{-1/2}
L = D @ A @ D
L_sym = I - D @ A @ D # normalised symmetric laplacian as 

# Spectral decomposition

res2_vals, res2_vecs = la.eigh(L_sym) # eigendecomposition 




def renorm(x):
    return x / np.sqrt(np.sum(x**2))


T = np.zeros((500,500))


for i in range(len(res2_vecs)):
    for j in range(len(res2_vecs)):
        
        T[i,j] = res2_vecs[i,j]/np.sqrt(np.sum(res2_vecs[i]**2))
        
        
    
    
    
Vnorm = np.apply_along_axis(renorm, 1, res2_vecs)
test = T - res2_vecs  


row_check = np.linalg.norm(Vnorm, axis=1, keepdims=True)
print(np.sum(row_check))

col_check = np.linalg.norm(Vnorm, axis=0, keepdims=True)
print(np.sum(col_check))



ALC_spec_GM = alc(np.corrcoef(res2_vecs),"GM")
num_clusters = np.max(ALC_spec_GM[0])
print(num_clusters)


# Adjusted Rand Index

ARI_GM = metrics.adjusted_rand_score(labels, sol_GM[0])
ARI_KM = metrics.adjusted_rand_score(labels, sol_KM[0])





''' ------------------------------------------------------------------------------------------------------------------------ '''


""" ALC Method """



''' 3 Concentric Circle Data Test Set'''
# Simulate 3 concentric circle data
def sim(N=500):
    r = np.random.choice([1, 100, 200], N, replace=True) + np.random.normal(0, 0.1, N)
    X = np.random.uniform(-1, 1, N)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
    return x, y, r

dat_X1, dat_X2, dat_r = sim()

# Data in X-space
X = np.column_stack((dat_X1, dat_X2))
plt.scatter(dat_X1, dat_X2, cmap='viridis')

corr_X = np.corrcoef(X)
plt.figure(figsize=(20,18))  # Optional: Set the size of the plot
sns.heatmap(corr_X, annot=False, cmap='viridis', fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
plt.show()


ALC_GM = alc(np.corrcoef(X),"GM")
plt.scatter(dat_X1, dat_X2, c = ALC_GM[0], s = 50, cmap='viridis')

ALC_KM = alc(np.corrcoef(X),"KM")
plt.scatter(dat_X1, dat_X2, c = ALC_KM[0], s = 50, cmap='viridis')
 

''' kmeans = KMeans(n_clusters=3, init='k-means++', n_init=3, max_iter=10000, random_state=42)
SL_kmeans = kmeans.fit(X)
labels = kmeans.labels_
plt.scatter(dat_X1, dat_X2, c = labels,  s = 50, cmap='viridis') '''


''' ------------------------------------------------------------------------------------------------------------------------ '''



""" Ng, Jordan and Weiss (2002) Spectral Clustering """

''' 3 Concentric Circle Data Test Set'''
# Simulate 3 concentric circle data
def sim(N=500):
    label = np.random.choice([1, 100, 200], N, replace=True)
    r = 2*label + np.random.normal(0, 0.1, N)
    X = np.random.uniform(-1, 1, N)
    x = r * np.cos(2 * np.pi * X)
    y = r * np.sin(2 * np.pi * X)
    return x, y, r, label

dat_X1, dat_X2, dat_r, dat_label = sim()
plt.scatter(dat_X1, dat_X2, cmap='viridis')


# Data in X-space
data = np.column_stack((dat_X1, dat_X2))
labelling = np.column_stack((dat_r, dat_label))

# First find the similarity matrix 
# Distance matrix calculation
def dist_mat(X):
    S = cdist(X, X, 'euclidean') # finds euclidean distance
    return S
res_S = dist_mat(data) 


####################  Finding optimal sigma ##################################

M = 50 
sigmas = np.linspace(0.05, 2, M)
cluster_quality = np.zeros(M)
I = np.eye(len(res_S))

for i,sig in enumerate(sigmas):
    
    A = np.exp(-res_S**2 / (2 * sig**2)) # Gaussian Affinity matrix
    D = np.diag(1 / np.sqrt(np.sum(A, axis=1))) # finding D^{-1/2}, axis=1 is summing across the rows. This is the degree matrix p.2 from Tut of SC
    #L = D @ A @ D # normalised symmetric laplacian (Ng, Jordan and Weiss)
    L_sym = I - D @ A @ D # normalised symmetric laplacian (Tutorial on spectral clustering)
    # L_rw  = 

    
    res2_vals, res2_vecs = la.eigh(L_sym) # eigendecomposition 
    #idx = res2_vals.argsort()[::-1] # finding indices for reordering due to SciPy eigh function output 
    #sorted_eigenvalues = res2_vals[idx] 
    #sorted_eigenvectors = res2_vecs[:, idx]
    Vnorm = res2_vecs
    
    ALC_spec_GM = alc(np.corrcoef(Vnorm),"GM")
    cluster_quality[i] = metrics.adjusted_rand_score(dat_label, ALC_spec_GM)

    
# Affinity matrix and Laplacian construction
max_ARI = np.max(cluster_quality)
print(max_ARI)
sigma = sigmas[np.argmax(cluster_quality)] # need to test a range of these for maximise/minimise some objective function
print(sigma)


sigma = 0.2


I = np.eye(len(res_S))
A = np.exp(-res_S**2 / (2 * sigma**2)) # Gaussian Affinity matrix
D = np.diag(1/np.sum(A,axis = 1)) #D^{-1}
col_sum = np.sum(A, axis = 1)
label_check = np.column_stack((col_sum, dat_label))
D_star = np.diag(np.sum(A, axis = 1))
D_inv  = np.linalg.inv(D_star)
diff = D_star - D_inv

#D = np.diag(1 / np.sqrt(np.sum(A, axis=1))) # finding D^{-1/2}
#L = D @ A @ D
#L_sym = I - D @ A @ D # normalised symmetric laplacian as 
L_rw = I  - D @ A


# Spectral decomposition

res2_vals, res2_vecs = la.eigh(L_rw) # eigendecomposition 

row_check = np.linalg.norm(res2_vecs, axis=1, keepdims=True)
print(np.sum(row_check))

col_check = np.linalg.norm(res2_vecs, axis=0, keepdims=True)
print(np.sum(col_check))

#vals = la.eigvals(L_sym)
#B_mat = vals @ 
#eigvects = la.solve(L_sym, )

#idx = res2_vals.argsort()[::-1] # finding indices for reordering due to SciPy eigh function output 
#sorted_eigenvalues = res2_vals[idx] 
#sorted_eigenvectors = res2_vecs[:, idx]

def renorm(x):
    return x / np.sqrt(np.sum(x**2))
Vnorm = np.apply_along_axis(renorm, 1, res2_vecs)


T = np.zeros((500,500))


for i in range(len(res2_vecs)):
    for j in range(len(res2_vecs)):
        
        T[i,j] = res2_vecs[i,j]/np.sqrt(np.sum(res2_vecs[i]**2))
        
        
    
    
    
test = np.round(T - res2_vecs,5) 


row_check = np.linalg.norm(Vnorm, axis=1, keepdims=True)
print(np.sum(row_check))

col_check = np.linalg.norm(Vnorm, axis=0, keepdims=True)
print(np.sum(col_check))




"""
The eigenvectors are already normalised

# Normalization function
def renorm(x):
    return x / np.sqrt(np.sum(x**2))

Vnorm = np.apply_along_axis(renorm, 1, V)

"""


G = res2_vecs.T @ res2_vecs



sub = Vnorm[:,[0,1,2]]

corr_V = np.corrcoef(res2_vecs) * 1000


ALC_spec_GM = alc(corr_V,"GM")
num_clusters = np.max(ALC_spec_GM)
#plt.scatter(dat_X1, dat_X2, c = ALC_spec_GM, cmap='viridis')
plt.scatter(data[:, 0], data[:, 1], c=ALC_spec_GM, s=50, cmap='viridis')
print(num_clusters)

val_check = np.round(res2_vals, 3)
values, counts = np.unique(val_check, return_counts = True)
round_corr = np.round(corr_V)

col_check = np.linalg.norm(round_corr, axis=0, keepdims=True)
print(np.sum(col_check))



plt.scatter(data[:, 0], data[:, 1], c=dat_label, s=50, cmap='viridis')
ARI_spec_GM = metrics.adjusted_rand_score(dat_label, ALC_spec_GM[0])

corr_X = np.corrcoef(data)
ALC_GM = alc(corr_X,"GM")
ARI_GM = metrics.adjusted_rand_score(dat_label, ALC_GM[0])




ALC_spec_KM = alc(np.corrcoef(Vnorm),"KM")
num_clusters = np.max(ALC_spec_KM[0])
plt.scatter(dat_X1, dat_X2, c = ALC_spec_KM[0], s = 50, cmap='viridis')
print(num_clusters)

