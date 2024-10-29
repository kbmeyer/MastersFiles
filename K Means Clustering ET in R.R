library(plot3D)
# A function to simulate three concentric circles:
sim = function(N = 500)
{
  r  = sample(1:3,N, replace   = TRUE)*2+rnorm(N,0,0.1) 
  X  = runif(N,-1,1)
  x  = r*cos(2*pi*X) 
  y  = r*sin(2*pi*X)
  return(list(X1 = x, X2 = y,r =r))
}
dat = sim()

# Data in X-space
X = cbind(dat$X1,dat$X2)


dist_mat = function(X)
{
  N     = dim(X)[1]
  ones  = matrix(1,N,1)
  S     = matrix(0,N,N)
  for(i in 1:N)
  {
    # Finding the distance between all the points in the "X" space, doing it row by row,
    S[i,] = sqrt(rowSums((ones%*%X[i,]-X)^2)) # vectorised distance function across all points in "X" space
    
  }
  return(list(S=S))
}

# Data is now on the similarity graph form
res = dist_mat(X)


sig = 0.2                       # Control parameter
A = exp(-res$S^2/(2*sig^2))     # Affinity matrix (I think this can be replaced.)
D = diag(1/sqrt(rowSums(A)))
L = D%*%A%*%D                   # Construct L

#Check:
#library(expm)
#sqrtm(diag(1/(rowSums(A))))[1:4,1:4]
#D[1:4,1:4]


# Eigen decomposition:
res2 = eigen(L)
k = 3                      
V = res2$vectors[,1:k]        # Pick k classes.
renorm = function(x){x/sqrt(sum(x^2))}
Vnorm = t(apply(V,1,renorm))  # Normalize vectors. 



# A function which explicitly delineates K-means. 
# - Runs set number of iterations
# - We can specify the initial clusters by index

# check if centroids are actual data points or new "point" that minimises the average distance between points in clusters

k_means = function(X,initial_index,iterations)
{
  Mus   = as.matrix(X[initial_index,]) # Centroids, set up with starting point of intial ccentroids 
  N     = dim(X)[1] #
  d     = dim(X)[2]
  K     = dim(as.matrix(Mus))[1] 
  ones  = matrix(1,N,1)
  dists = matrix(0,N,K)
  error = c()
  for(i in 2:iterations) # since there is no convergence criterion we do it for a set amount of iterations
  {
    dists = dists*0 # resetting to zero at start of a new iteration
    for(j in 1:K) # K is number of clusters so setting up K centroids
    {
      dists[,j] = rowSums((X-ones%*%Mus[j,])^2) # Euclidean diff. between the first centriod and all data points
    }
    assigned_labels = apply(dists,1,which.min) # finding which centroid is closest to each data point. 
    error[i-1] = sum(apply(dists,1,min)) # overall distance between each centroid and each data point in it's "cluster"
    #for(k in 1:K)
    #{
    #    wh = which(assigned_labels == k)
    #    Mus[k,] = colSums(as.matrix(X[wh,]))/length(wh)s
    #}
    for(k in 1:K)
    {
      if(any(assigned_labels == k)) 
      {
        wh      = which(assigned_labels == k) #index for which ones are part of which "cluster"
        Mus[k,] = colSums(as.matrix(X[wh,]))/length(wh) # finding the mean point of the "cluster"
      }else{ # if not the particular cluster that k is specifying, then we take find the max index and allocate the centroid to 
             # to that area of the "X" Space. This is potentially where the other "clusters" are. 
        wh_rep  = which.max(apply(dists,1,max))
        Mus[k,] = X[wh_rep,]
      }
    }
  }
  return(list(centroids = Mus,assigned_labels = assigned_labels,dists = dists,K= K,error = error))
}
# This is the trick for picking good starting values for K-Means:
DV = dist_mat(Vnorm)
i1 = 1
i2 = which.max(DV$S[i1,])
i3 = which.max(DV$S[i1,]*DV$S[i2,])

# Now cluster on the chosen eigenvectors:
res_clust = k_means(Vnorm,c(i1,i2,i3),20)

# Plot the original data, the assigned clusters, and eigenvectors with the
# cluster centroids:
par(mfrow = c(2,2))
plot(dat$X2~dat$X1, pch = 16,cex = 1,main = 'Observations')
plot(X[,2]~X[,1],col = res_clust$assigned_labels+1,cex = 1, pch =16,main = 'Assigned Labels')

library(plot3D)
N = dim(Vnorm)[1]
jitter = 0.1   # Perturb observations to make the `clumps' visible. 
scatter3D(Vnorm[,1]+runif(N)*jitter,Vnorm[,2]+runif(N)*jitter,Vnorm[,3]+runif(N)*jitter, pch =16,main = 'Eigen Vectors')
points3D(res_clust$centroids[,1],res_clust$centroids[,2],res_clust$centroids[,3],pch = 4,col = 'black',add = TRUE,cex = 3)


M      = 50                      # Search through so many values
sigmas = seq(0.05,2,length = M)  # Sequence of values
cluster_dispersion = rep(NA,M)   # Save scores here
for(i in 1:length(sigmas))
{
  
  sig = sigmas[i]                 # Control parameter
  A = exp(-res$S^2/(2*sig^2))     # Affinity matrix (I think this can be replaced.)
  D = diag(1/sqrt(rowSums(A)))    # Diagonal Matrix 
  L = D%*%A%*%D                   # Construct L
  
  # Eigen decomposition:
  res2 = eigen(L)
  k = 3                      
  V = res2$vectors[,1:k]        # Pick k classes.
  Vnorm = t(apply(V,1,renorm))  # Normalize vectors. 
  
  DV = dist_mat(Vnorm)          # Distance 
  i1 = 1                        # ET's choices of an initialisation
  i2 = which.max(DV$S[i1,])
  i3 = which.max(DV$S[i1,]*DV$S[i2,])
  
  # Now cluster on the chosen eigenvectors:
  res_clust = k_means(Vnorm,c(i1,i2,i3),20)
  
  cluster_dispersion[i] = rev(res_clust$error)[1] # smallest error for that iteration, not sure why 
  # why we call it cluster dispersion
}


par(mfrow = c(1,1))
plot(cluster_dispersion~sigmas,type =  'b',pch = 16)
abline(h=0,v=0,lty = 3)


par(mfrow = c(3,2))

# Now running algo for set of sigmas
sigmas = c(0.05,0.2,1)
for(i in 1:3)
{
  sig = sigmas[i]                       # Control parameter
  A = exp(-res$S^2/(2*sig^2))     # Affinity matrix (I think this can be replaced.)
  D = diag(1/sqrt(rowSums(A)))
  L = D%*%A%*%D                   # Construct L
  
  # Eigen decomposition:
  res2 = eigen(L)
  k = 3                      
  V = res2$vectors[,1:k]        # Pick k classes.
  Vnorm = t(apply(V,1,renorm))  # Normalize vectors. 
  
  DV = dist_mat(Vnorm)
  i1 = 1
  i2 = which.max(DV$S[i1,])
  i3 = which.max(DV$S[i1,]*DV$S[i2,])
  
  # Now cluster on the chosen eigenvectors:
  res_clust = k_means(Vnorm,c(i1,i2,i3),20)
  expr = substitute(sigma == a,list(a=sig))
  plot(X[,2]~X[,1],col = res_clust$assigned_labels+1,cex = 1, pch =16,main = expr)
  jitter = 0.1   # Perturb observations to make the `clumps' visible. 
  scatter3D(Vnorm[,1]+runif(N)*jitter,Vnorm[,2]+runif(N)*jitter,Vnorm[,3]+runif(N)*jitter, pch =16,main = 'Eigen Vectors')
  points3D(res_clust$centroids[,1],res_clust$centroids[,2],res_clust$centroids[,3],pch = 4,col = 'black',add = TRUE,cex = 3)
}