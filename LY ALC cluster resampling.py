# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:28:23 2019

@author: Lionel Yelibi
Cluster resampling scheme, See Section V 
Agglomerative Likelihood Clustering (2019,2020,2021) See pre-print: https://arxiv.org/abs/1908.00951

"""

import numpy as np
import itertools
from sklearn import metrics
from timeseries_generator import onefactor
from function_aspcv3 import alc

''' onefactor_timeries is a function which generates a data-set of synthetic
correlated timeseries
https://arxiv.org/abs/cond-mat/0101237
https://arxiv.org/abs/1908.00951


'''

import numpy as np

def onefactor_timeseries(N, C,L,gs = .8, model='normal', mu=0):
    """Returns a data-set of correlated and clustered timeseries.
    
       :param N: Number of time-series
       :param C: Number of clusters.
       :param L: Time-series length.
       :param gs: coupling parameter.
       :param model: time-series distribution, a string in ['model','student']
       
       :return: the data set of timeseries, and the cluster membership key.
    """
    key = np.sort(np.random.choice(range(C),N))

    if isinstance(gs, np.ndarray):
        gsvector = gs[key]
    elif isinstance(gs, list):
        gs = np.array(gs)
        gsvector = gs[key]

    else:
        gsvector =gs*np.ones(N)
        


    '''one factor model requires:
        eta as the cluster random variable
        epsilon as the object random variable.
        We allow for the selection of gaussian or student-t models.
        i.e. stock market returns have fat tails, and aren't gaussian'''
    if model == 'normal':
        eta = np.random.normal(loc=mu,scale=1,size=(C,L))
        epsilon = np.random.normal(loc=0,scale=1,size = (N,L))
    elif model =='student':
        eta = np.random.standard_t(2.5,size = (C,L))
        epsilon = np.random.standard_t(2.5,size = (N,L))
    
    

    ''' coupling paramater gs: varies from 0 to 1, and determines how much
    an object's features are tied to the cluster. Clusters where gs = 0 are very
    low density, and very noisy, whereas gs->1 means more correlated clusters'''    
    ''' 1-factor model: as seen in Giada-Marsili 2001'''
    
    gsvector = gsvector.reshape(-1,1)

    return gsvector*eta[key]+np.sqrt(1-gsvector**2)*epsilon, key
    
def multifactor_series(N, C, L, a=5,b=1):
    """
    Generates a simulated multi-factor time series dataset.
    
    This function simulates a dataset arising from a multi-factor model, where
    observations are driven by a combination of latent factors and random noise.
    
    Args:
        N (int): Number of observations (time points) in the series.
        C (int): Number of categories (or groups).
        L (int): Number of latent factors.
        a (float, optional): Shape parameter 'a' of the Beta distribution. Defaults to 5.
        b (float, optional): Shape parameter 'b' of the Beta distribution. Defaults to 1.
    
    Returns:
        tuple: A tuple containing the following elements: 
            X (np.ndarray): The simulated time series data (N x L).
            y (np.ndarray): Category assignments for each observation (N,).
            G (np.ndarray): Group membership matrix (N x C).
            F (np.ndarray): Latent factor matrix (C x L).
    """
    G = np.zeros((N, C))
    y = np.sort(np.random.randint(0, C, N))
    coef = np.random.beta(a, b, N)
    G[np.arange(N), y] = coef
        
    remaining_vals = 1 - G[np.arange(N), y]
    all_column_indices = np.arange(C).reshape(1, -1).repeat(N, axis=0) 
    mask = all_column_indices != y[:, np.newaxis]
    remaining_col_indices = all_column_indices[mask] 
    
    dirichlet_samples = np.random.dirichlet(np.ones(C - 1), size=N) * remaining_vals[:, np.newaxis]
    G[np.repeat(np.arange(N),C-1), remaining_col_indices] = dirichlet_samples.flatten()
    G = np.around(G,2)
    

    F = np.random.normal(0,1,(C,L))

    X = np.matmul(G,F)+ np.random.normal(0,0.05,(N,L))
    
    return X, y, G, F

def multifactorcollinear(n_class=3,
                         class_sizes=[4000, 4000, 4000],
                         n_groups=4,
                         group_sizes=[2, 20, 2, 20],
                         signal_strength=[.95, 0.5, 0.95, 0.5],
                         class_separability=[0.7, 0.7, 0.4, 0.40],):
    '''
    

    Parameters
    ----------
    n_class : TYPE, optional
        DESCRIPTION. The default is 3.
    class_sizes : TYPE, optional
        DESCRIPTION. The default is [4000, 4000, 4000].
    n_groups : TYPE, optional
        DESCRIPTION. The default is 4.
    group_sizes : TYPE, optional
        DESCRIPTION. The default is [2, 20, 2, 20].
    signal_strength : TYPE, optional
        DESCRIPTION. The default is [.95, 0.5, 0.95, 0.5].
    class_separability : TYPE, optional
        DESCRIPTION. The default is [0.7, 0.7, 0.4, 0.40].

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    fmats : TYPE
        DESCRIPTION.
    gmats : TYPE
        DESCRIPTION.
    feat_labels : TYPE
        DESCRIPTION.

    '''

    y = np.concatenate([[i]*class_sizes[i] for i in range(n_class)])
    feat_labels = np.concatenate([[i]*group_sizes[i] for i in range(n_groups)])

    n_obj = sum(class_sizes)
    xmats = []
    gmats = []
    fmats = []

    for g in range(n_groups):

        '''Factor Loading Matrix'''
        G = np.zeros((n_obj, n_class))
        G[np.arange(n_obj), y] = class_separability[g]

        all_column_indices = np.arange(n_class).reshape(1, -1).repeat(n_obj, axis=0)
        mask = all_column_indices != y[:, np.newaxis]
        remaining_col_indices = all_column_indices[mask]

        ''' Random factor loading for the remaining classes'''
        # remaining_vals = 1 - G[np.arange(n_obj), y]
        # dirichlet_samples = np.random.dirichlet(np.ones(n_class - 1), size=n_obj) * remaining_vals[:, np.newaxis]
        # G[np.repeat(np.arange(n_obj), n_class-1), remaining_col_indices] = dirichlet_samples.flatten()
        ''' Equal factor loading for the remaining classes'''
        # G[np.repeat(np.arange(n_obj), n_class-1), remaining_col_indices] = (1 - class_separability[g])/(n_class-1)
        G[np.repeat(np.arange(n_obj), n_class-1), remaining_col_indices] = 0.5
        
        # G = G/ np.linalg.norm(G, 1, axis=1).reshape(-1,1)
        G = G/ np.sum(G, axis=1).reshape(-1,1)

        ''' Factor matrix '''
        F = np.random.normal(0, 1, (n_class, group_sizes[g]))
        ''' Group Synthetic Data'''
        X = signal_strength[g]*np.matmul(G, F) + np.sqrt(1-signal_strength[g]**2)*np.random.normal(0, 1,(n_obj, group_sizes[g]))
        xmats.append(X)
        fmats.append(F)
        gmats.append(G)
    data = np.concatenate(xmats, axis=1)
    return data, y, fmats, gmats, feat_labels

def generate_twoclassproblems(y, N, L, gs_0=0.9, gs_1=0.9):
    '''
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    gs_0 : TYPE, optional
        DESCRIPTION. The default is 0.9.
    gs_1 : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    '''
    c_0 = np.random.normal(0, 1, L)
    c_1 = np.random.normal(0, 1, L)

    epsi = np.random.normal(0, 1, (N, L))

    x = (1-y)*(gs_0*c_0 + np.sqrt(1-gs_0**2)*epsi) + y*(gs_1*c_1 + np.sqrt(1-gs_1**2)*epsi)
    return x


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
# plt.tight_layout()