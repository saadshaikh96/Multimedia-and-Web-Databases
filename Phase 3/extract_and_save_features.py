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
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-m', required=True, type=str, help='method')

def get_label(labeled_images_folder, method):
    extract_and_save_features(labeled_images_folder, method)
    # data, ids = load_saved_features(labeled_images_folder, method)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    labeled_folder = args.f
    method = args.m

    get_label(labeled_folder, method)
