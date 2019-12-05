import os
import math
import argparse
import numpy as np
from feature_extractor import get_data_matrix, extract_and_save_features, load_saved_features
from pca import get_pca_decomposition, get_pca_transform
from svd import get_svd_decomposition, get_svd_transform
from utils import get_images_by_metadata
from kmeans import get_cluster_centers
from kmeans2 import get_centroids
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using latent semantics')
parser.add_argument('-c', type=int, default=5, help='The number of clusters to generate')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder where unlabeled images are stored')


def classify_item(item, dorsal_centroids, palmar_centroids, c):
    distances = []
    for i in dorsal_centroids:
        dist = euclidean(i, item)
        distances.append(("dorsal", dist))
    for i in palmar_centroids:
        dist = euclidean(i, item)
        distances.append(("palmar", dist))
    distances = sorted(distances, key = lambda x: x[1])
    palmar_count = 0
    dorsal_count = 0
    for dist in distances[:max(3, int(c/2))]:
        if dist[0] == "dorsal":
            dorsal_count += 1
        else:
            palmar_count += 1
    if dorsal_count > palmar_count:
        return "dorsal"
    else:
        return "palmar"

def get_label(labeled_images_folder, unlabeled_images_folder, c):
    """
    Given a folder with unlabeled images, the system labels
    them as dorsal or palmer using clustering
    :param folder: The folder containing images for training
    :param c: The number of clusters to generate
    :return: the labels for the unlabeled images
    """
    k = 20

    # Extract features for the input folder
    features_data, object_ids = get_data_matrix(labeled_images_folder, method="cm")

    # PCA decomposition
    u, vt = get_svd_decomposition(features_data, k)

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = get_data_matrix(unlabeled_images_folder, method="cm")

    # Get dorsal images
    dorsal_data, dorsal_image_ids = get_images_by_metadata(object_ids, u, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, palmar_image_ids = get_images_by_metadata(object_ids, u, labeled_images_folder, dorsal=0)

    # transform ul_features
    ul_features_u = get_svd_transform(vt, ul_features_data)

    # Get clusters for dorsal
    best_dorsal_centroids = None
    best_palmar_centroids = None
    max_correct_count = None
    for _ in range(100):
        dorsal_centroids = get_cluster_centers(dorsal_data, c, 10)
        # dorsal_centroids = get_centroids(dorsal_data, c, 10)
        # Get clusters for palmar
        palmar_centroids = get_cluster_centers(palmar_data, c, 10)
        # palmar_centroids = get_centroids(palmar_data, c, 10)
        correct_count = 0
        for item in dorsal_data:
            label = classify_item(item, dorsal_centroids, palmar_centroids, c)
            if label == 'dorsal':
                correct_count += 1
        for item in palmar_data:
            label = classify_item(item, dorsal_centroids, palmar_centroids, c)
            if label == 'palmar':
                correct_count += 1
        if max_correct_count is None or max_correct_count < correct_count:
            max_correct_count = correct_count
            best_dorsal_centroids = dorsal_centroids
            best_palmar_centroids = palmar_centroids

    dorsal_centroids = best_dorsal_centroids
    palmar_centroids = best_palmar_centroids

    # Visualize clusters
    visualize_clusters(dorsal_data, dorsal_image_ids, dorsal_centroids, labeled_images_folder)
    visualize_clusters(palmar_data, palmar_image_ids, palmar_centroids, labeled_images_folder)

    # Only to measure accuracy
    _, ul_dorsal_image_ids = get_images_by_metadata(ul_object_ids, ul_features_u, unlabeled_images_folder, dorsal=1)

    _, ul_palmar_image_ids = get_images_by_metadata(ul_object_ids, ul_features_u, unlabeled_images_folder, dorsal=0)

    # Label images
    pred_true_count = 0
    for features, object_id in zip(ul_features_u, ul_object_ids):
        distances = []
        for i in dorsal_centroids:
            dist = euclidean(i, features)
            distances.append(("dorsal", dist))
        for i in palmar_centroids:
            dist = euclidean(i, features)
            distances.append(("palmar", dist))
        distances = sorted(distances, key = lambda x: x[1])
        palmar_count = 0
        dorsal_count = 0
        for dist in distances[:max(3, int(c/2))]:
            if dist[0] == "dorsal":
                dorsal_count += 1
            else:
                palmar_count += 1
        if dorsal_count > palmar_count:
            if object_id in ul_dorsal_image_ids:
                pred_true_count += 1
            print ("Image %s is dorsal." % object_id)
        else:
            if object_id in ul_palmar_image_ids:
                pred_true_count += 1
            print ("Image %s is palmar." % object_id)

    print ("accuracy is %s" % (pred_true_count*100/len(ul_object_ids)))

def visualize_clusters(data, data_ids, centroids, labeled_images_folder):
    centers_count = len(centroids)
    classify = []
    for _ in range(centers_count):
        classify.append([])
    for data_id, item in zip(data_ids, data):
        index = 0
        centroid_dist = []
        for centroid in centroids:
            dist = euclidean(centroid, item)
            centroid_dist.append((index, dist))
            index += 1
        centroid_dist = sorted(centroid_dist, key = lambda x: x[1])
        classify[centroid_dist[0][0]].append(data_id)
    for item in classify:
        plot_images(item, labeled_images_folder)

def plot_images(images, folder_path):
    total_plots = len(images)
    cols = rows = math.ceil(math.sqrt(total_plots))
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    index = 0
    for row in axes:
        for ax in row:
            if index < len(images):
                cluster_image = os.path.join(folder_path, images[index])
                filename = images[index]
                img = mpimg.imread(cluster_image)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(filename)
                index += 1
            else:
                ax.axis('off')
    plt.show()

def euclidean(x,y):
    return np.linalg.norm(x-y)

if __name__ == "__main__":
    args = parser.parse_args()
    c = args.c
    labeled_folder = args.f
    unlabeled_folder = args.u

    get_label(labeled_folder, unlabeled_folder, c)
