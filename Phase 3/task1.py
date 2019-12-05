import argparse
import numpy as np
from feature_extractor import get_data_matrix
from pca import get_pca_decomposition, get_pca_transform
from utils import get_images_by_metadata

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using latent semantics')
parser.add_argument('-k', type=int, default=5, help='The number of latent semantics to generate')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder where unlabeled images are stored')


def get_label(labeled_images_folder, unlabeled_images_folder, k):
    """
    Given a folder with unlabeled images, the system labels
them as dorsal or palmer
    :param folder: The folder containing images for training
    :param k: The number of latent dimensions to compute
    :return: the labels for the unlabeled images
    """
    # Extract features for the input folder
    features_data, object_ids = get_data_matrix(labeled_images_folder, method="cm")

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = get_data_matrix(unlabeled_images_folder, method="cm")

    # Get dorsal images
    dorsal_data, _ = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, _ = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=0)

    # PCA decomposition
    dorsal_u, dorsal_vt = get_pca_decomposition(dorsal_data, k)
    palmar_u, palmar_vt = get_pca_decomposition(palmar_data, k)


    dorsal_u_min = dorsal_u.min(axis=0)
    palmar_u_min = palmar_u.min(axis=0)

    mean_dorsal = np.mean(dorsal_u, axis=0)
    mean_palmar = np.mean(palmar_u, axis=0)

    max_dorsal_dist = None
    max_palmar_dist = None
    zeros = np.asarray([0]*dorsal_u.shape[1])
    for item in dorsal_u:
        dist = euclidean(zeros, item)    
        if not max_dorsal_dist or dist > max_dorsal_dist:
            max_dorsal_dist = dist

    for item in palmar_u:
        dist = euclidean(zeros, item)    
        if not max_palmar_dist or dist > max_palmar_dist:
            max_palmar_dist = dist

    # Only to measure accuracy
    _, ul_dorsal_image_ids = get_images_by_metadata(ul_object_ids, ul_features_data, unlabeled_images_folder, dorsal=1)

    _, ul_palmar_image_ids = get_images_by_metadata(ul_object_ids, ul_features_data, unlabeled_images_folder, dorsal=0)

    # transform ul_features
    sems_in_dorsal_space = get_pca_transform(dorsal_vt, ul_features_data)
    sems_in_palmar_space = get_pca_transform(palmar_vt, ul_features_data)

    mean_dorsal = (mean_dorsal - dorsal_u_min)/max_dorsal_dist
    mean_palmar = (mean_palmar - palmar_u_min)/max_palmar_dist

    pred_true_count = 0
    for i in range(ul_features_data.shape[0]):
        normalized_point_dorsal = (sems_in_dorsal_space[i] - dorsal_u_min)/max_dorsal_dist
        dorsal_normal_dist = euclidean(normalized_point_dorsal, mean_dorsal)

        normalized_point_palmar = (sems_in_palmar_space[i] - palmar_u_min)/max_palmar_dist
        palmar_normal_dist = euclidean(normalized_point_palmar, mean_palmar)

        if dorsal_normal_dist <= palmar_normal_dist:
            if ul_object_ids[i] in ul_dorsal_image_ids:
                pred_true_count += 1
            print ("%s is dorsal" % ul_object_ids[i])
        else:
            if ul_object_ids[i] in ul_palmar_image_ids:
                pred_true_count += 1            
            print ("%s is palmar" % ul_object_ids[i])

    print ("accuracy is %s" % (pred_true_count*100/len(ul_object_ids)))

def euclidean(x,y):
    return np.linalg.norm(x-y)

if __name__ == "__main__":
    args = parser.parse_args()
    k = args.k
    labeled_folder = args.f
    unlabeled_folder = args.u

    #get_label("phase3_sample_data/Labelled/Set2", "phase3_sample_data/Unlabelled/Set2", 30)
    get_label(labeled_folder, unlabeled_folder, k)

