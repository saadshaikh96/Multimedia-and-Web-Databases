# python task4.py -k 5 -f phase3_sample_data/Labelled/Set2 -u phase3_sample_data/Unlabelled/Set2 -c dtree

import argparse
from utils import get_images_by_metadata
import numpy as np

import dtree_temp
import ppr_classifier
import feature_extractor
import svm

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using classifiers')
parser.add_argument('-k', type=int, default=5, help='PPR - number of edges')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder where unlabeled images are stored')
parser.add_argument('-c', required=True, type=str, help='classifier type: dtree/svm/ppr')
excel_sheet = "HandInfo.xlsx"


def dtree(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids, unlabeled_folder=None):
    dorsal_labels = np.array([[1]]*len(dorsal_image_ids))
    palmar_labels = np.array([[0]]*len(palmar_image_ids))
    dorsal_train = np.hstack((dorsal_data, dorsal_labels))
    palmar_train = np.hstack((palmar_data, palmar_labels))

    training_data = np.vstack((dorsal_train, palmar_train))
    # print(ul_data, ul_data.shape)
    print('########################   Training   ##############################')

    my_tree = dtree_temp.build_tree(training_data)
    print('########################   Trained   ##############################')
    predictions = list()
    for ul_vec, ul_id in zip(ul_data, ul_image_ids):
        # print(ul_id, ul_vec.shape)
        prediction = dtree_temp.classify(ul_vec, my_tree)
        for key, val in prediction.items():
            predictions.append(key)
    print('#####################   Finished   #################################')

    # for task 6, there is ground truth. So dont measure accuracies
    if not unlabeled_folder:
        return predictions
    
    measure_accuracy(ul_image_ids, ul_data, unlabeled_folder, predictions)


def measure_accuracy(ul_image_ids, ul_data, unlabeled_folder, predictions):
    # Get dorsal and palmar images
    _, dorsal_test_ids = get_images_by_metadata(ul_image_ids, ul_data, unlabeled_folder, dorsal=1)
    _, palmar_test_ids = get_images_by_metadata(ul_image_ids, ul_data, unlabeled_folder, dorsal=0)
    # how the images are classified will be stored in the following lists
    labels = list()
    correct, incorrect = 0, 0
    for prediction, hand in zip(predictions, ul_image_ids):
        labels.append([hand, prediction])
        if prediction == 1:
            print ("%s is dorsal." % hand)
        else:
            print ("%s is palmar." % hand)
        if labels[-1][1] == 1 and labels[-1][0] in dorsal_test_ids:
            correct += 1
        elif labels[-1][1] == 0 and labels[-1][0] in palmar_test_ids:
            correct += 1
        else:
            incorrect += 1
    print('Accuracy =', correct*100/(correct+incorrect))
    return labels


def get_label(labeled_images_folder, unlabeled_images_folder, k, method="hog"):
    """
    Given a folder with unlabeled images, the system labels
them as dorsal or palmer
    :param folder: The folder containing images for training
    :param k: The number of latent dimensions to compute
    :return: the labels for the unlabeled images
    """
    # Extract features for the input folder
    features_data, object_ids = feature_extractor.get_data_matrix(labeled_images_folder, method)

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = feature_extractor.get_data_matrix(unlabeled_images_folder, method)

    # Get dorsal images
    dorsal_data, dorsal_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, palmar_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=0)

    unlabelled_data, unlabelled_image_ids = get_images_by_metadata(ul_object_ids, ul_features_data, unlabeled_images_folder)

    return dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, unlabelled_data, unlabelled_image_ids


def classify(labeled_folder, unlabeled_folder, k, classifier):
    """

    :param labeled_folder:
    :param unlabeled_folder:
    :param k:
    :param classifier:
    :return:
    """
    # TODO  write your code here
    dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, ul_data, ul_image_ids = get_label(labeled_folder, unlabeled_folder, k, method="cm")

    if classifier == 'dtree':
        # print('Unlabeled images, labels, confidence')
        preds = dtree(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids, unlabeled_folder)
        return preds
    elif classifier == "svm":
        preds = svm.do_svm(dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, ul_data, ul_image_ids)
        return preds
    elif classifier == 'ppr':
        preds, confs = ppr_classifier.ppr(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids, unlabeled_folder, k)
        measure_accuracy(ul_image_ids, ul_data, unlabeled_folder, preds)
        return preds

if __name__ == "__main__":
    args = parser.parse_args()
    k = args.k
    l_folder = args.f
    u_folder = args.u
    classifier = args.c
    # l_folder = 'phase3_sample_data/Labelled/Set2'
    # u_folder = 'phase3_sample_data/Unlabelled/Set 2'
    # k = 3
    # classifier = 'svm'
    preds = classify(l_folder, u_folder, k, classifier)
    #accuracy = preds,
    #    classify(l_folder, u_folder, k, classifier)
