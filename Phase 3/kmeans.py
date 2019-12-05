import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from random import randint

threshold = 0.0001

def Euclidean(x,y):
    return np.linalg.norm(x-y)

# def l1_norm(x, y):
#     sum1 = 0
#     for i in range(len(x)):
#         sum1 += abs(x[i]-y[i])
#     return sum1
#
# def areDifferentCentroids(a,b):
#     if(len(a)!=len(b)):
#         return 1
#     for i in range(0,len(a)):
#         dist = l1_norm(a[i], b[i])
#         if dist > threshold:
#             return 1
#     return 0

def areDifferentCentroids(a,b):
    if(len(a)!=len(b)):
        return 1
    for i in range(0,len(a)):
        if(a[i] != b[i]):
            return 1
    return 0

def findMaxLengthCluster(cluster_list):
    max_index = 0
    for jj in range(len(cluster_list)):
        if(len(cluster_list[jj]) > len(cluster_list[max_index])):
            max_index = jj

    return max_index

def get_cluster_centers(data, k, r):
    data = np.asarray(data)
    data = data.tolist()
    new_centroids = [[] for xx in range(k)]
    old_centroids = []
    for i in range(k):
        old_centroids.append(data[np.random.randint(0,len(data))])

    count = 0
    best_centers = None
    best_SSE = None
    for _ in range(r):
        while(areDifferentCentroids(new_centroids,old_centroids)):
            cluster_points = [[] for xx in range(k)]
            if(count>0):
                old_centroids = new_centroids
                new_centroids = [[] for xx in range(k)]

            count += 1

            for data_point in range(len(data)):
                min_dist = 99999999
                min_center = 0
                for iterator,center in enumerate(old_centroids):
                    distance = Euclidean(np.array(data[data_point]), np.array(center))

                    if(distance < min_dist):
                        min_dist = distance
                        min_center = iterator
                cluster_points[min_center].append(data_point)

            #Shift samples from max cluster to empty cluster
            for ite,cluster in enumerate(cluster_points):
                if(not cluster):
                    max_len_cluster_index = findMaxLengthCluster(cluster_points)
                    move_samples = np.random.randint(1,len(cluster_points[max_len_cluster_index]))
                    for mm in range(move_samples):
                        sample_to_move = cluster_points[max_len_cluster_index][0]
                        cluster_points[ite].append(sample_to_move)
                        cluster_points[max_len_cluster_index].remove(sample_to_move)

            #Calculating new centroids
            for ii in range(len(cluster_points)):
                temp_data = []
                for jj in cluster_points[ii]:
                    temp_data.append(data[jj])

                temp_data = np.array(temp_data)
                temp_data = np.average(temp_data, axis=0)
                new_centroids[ii] = temp_data.tolist()
        SSE = 0
        for data_point in range(len(data)):
            min_dist = 99999999
            min_center = 0
            for iterator,center in enumerate(new_centroids):
                distance = Euclidean(np.array(data[data_point]), np.array(center))

                if(distance < min_dist):
                    min_dist = distance
                    min_center = iterator
            SSE += min_dist
        if best_SSE is None or best_SSE > SSE:
            best_SSE = SSE
            best_centers = new_centroids
    return best_centers
