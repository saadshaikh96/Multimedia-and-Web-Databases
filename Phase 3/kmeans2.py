# python 3.5
# Place Dataset_1.txt in the same folder and run q1.py K r
import numpy as np
import matplotlib.pyplot as plt
import sys, ast

#distance function
def distance(a,b,axis=1):
    return np.linalg.norm(a - b, axis=axis)

def K_clusters(centers, data, K):
    clusters = np.zeros(data.shape[0])
    better_center_found = True
    ii = 0
    SSE_all = []
    while better_center_found:
        ii += 1
        SSE = 0
        centers_new = np.zeros(centers.shape)
        for i in range(len(data)):
            distances = distance(data[i],centers)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        for i in range(K):
            cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
            if not cluster_points:
                centers_new[i] = centers[i]
            else:
                centers_new[i] = np.mean(cluster_points, axis=0)
                distances = distance(centers_new[i], cluster_points)
                SSE += np.sum(distances)
        if (distance(centers_new, centers, None) == 0):
            better_center_found = False
        centers = centers_new
        SSE_all.append(SSE/data.shape[0])
    return centers_new, SSE_all
	
def get_centroids(data, K, r):
    data = np.asarray(data)
    # D is the dimension of each sample
    D = data.shape[1]

    # Find mean and Std Deviation to initialize the centers
    mean_of_data = np.mean(data, axis=0)
    std_of_data = np.std(data, axis=0)

    best_SSE_all = None
    best_centers = None
    for item in range(r):
        centers = np.random.randn(K,D)*std_of_data + mean_of_data
        centers_new, SSE_all = K_clusters(centers, data, K)
        if best_SSE_all is None or best_SSE_all[-1] > SSE_all[-1]:
            best_SSE_all = SSE_all
            best_centers = np.array(centers_new, copy=True)
    return best_centers