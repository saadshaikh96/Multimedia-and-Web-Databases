# python task4.py -k 5 -f phase3_sample_data/Labelled/Set2 -u phase3_sample_data/Unlabelled/Set2 -c dtree

import argparse
from utils import get_images_by_metadata
import numpy as np
# from feature_extractor import get_hog_features_by_image_path
# import task3
import pandas as pd
import dtree_temp
import scipy
from task3 import sort_list
import feature_extractor
# from task3 import get_similar_images_for_each
order_file = 'order.csv'

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using latent semantics')
parser.add_argument('-k', type=int, default=3, help='The number of nearest neighbors')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder containing the unlabled images')
parser.add_argument('-c', required=True, type=str, help='the classifier to use')
excel_sheet = "HandInfo.xlsx"
vectors_file = 't4_vecs.csv'


def get_similar_images_for_each(lbp_outputs):
    objs = np.ones((len(lbp_outputs), len(lbp_outputs)))
    for i in range(len(lbp_outputs)):
        i_vec = np.array(lbp_outputs[i])
        i_mag = np.linalg.norm(i_vec)
        for j in range(i + 1, len(lbp_outputs)):
            j_vec = np.array(lbp_outputs[j])
            j_mag = np.linalg.norm(j_vec)
            objs[i, j] = objs[j, i] = (i_vec @ j_vec.T) / (i_mag * j_mag)
    return objs

#
# def find_similarity_old(cm_outputs, img_order):
#     """
#     Given a features of images with images, the similarity matrix is calculated
#     :param folder: list containing the names of images
#     :return: saves the three files with order of rows, similarity matrix, most similar images for each image
#     """
#     # TODO  write your code here
#
#     cm_outputs = np.array(cm_outputs)
#     print(cm_outputs.shape)
#     # cm_outputs contains a list of images and their corresponding feature vectors
#     obj = get_similar_images_for_each(cm_outputs)
#     # obj contains the object object distance matrix
#     similarity_order = np.array([])
#     for i in range(len(img_order) - 1):
#         itemp = obj[i, :]
#         # print(sort_list(img_order, itemp))
#         similarity_order = np.append(similarity_order, np.array(task3.sort_list(img_order, itemp)))
#     l = int(np.ceil(np.sqrt(len(similarity_order))))
#     similarity_order = similarity_order.reshape((l, len(similarity_order) // l))
#     return obj, similarity_order, img_order
#     # # print(similarity_order.shape)
#     # pd.DataFrame(obj).to_csv(vectors_file)
#     # pd.DataFrame(similarity_order).to_csv(similarity_order_file)
#     # pd.DataFrame(img_order).to_csv(order_file)


def find_similarity(lbp_outputs, img_order):
    """
    Given a folder with images, the pageranks are visualized
    :param folder: Folder containing the images
    :return: saves the three files with order of rows, similarity matrix, most similar images for each image
    """
    # # TODO  write your code here
    # lbp_outputs = []
    # img_order = []
    # for pic in os.listdir(folder):
    #     img_order.append(pic)
    #     temp = np.array(get_lbp_features_by_image_path(folder, pic))
    #     temp = temp.ravel()
    #     lbp_outputs.append([pic, temp])
    lbp_outputs = np.array(lbp_outputs)
    # lbp_outputs contains a list of images and their corresponding feature vectors
    obj = get_similar_images_for_each(lbp_outputs)
    # obj contains the object object distance matrix
    similarity_order = np.array([])
    for i in range(len(img_order) - 1):
        itemp = obj[i, :]
        # print(sort_list(img_order, itemp))
        similarity_order = np.append(similarity_order, np.array(sort_list(img_order, itemp)))
    l = int(np.ceil(np.sqrt(len(similarity_order))))
    similarity_order = similarity_order.reshape((l, len(similarity_order) // l))
    # print(similarity_order.shape)
    return obj, similarity_order, img_order
    # pd.DataFrame(obj).to_csv(vectors_file)
    # pd.DataFrame(similarity_order).to_csv(similarity_order_file)
    pd.DataFrame(img_order).to_csv(order_file)


def pagerank(obj, similarity_order, order, dorsal_count, palmar_count, k=9, p=0.85, personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph

    Inputs:
    -------
    Tg: the train matrix including dorsal and palmar vectors
    k: number of connections from each node
    p: damping factor
    personlize: if not None, should be an array with the size of the nodes containing probability distributions.
                It will be normalized automatically.
    reverse: If true, returns the reversed-PageRank

    Returns:
    --------
    PageRank Scores for the nodes

    """
    # order = np.array(pd.read_csv(order_file).as_matrix())
    dorsals = [1]*dorsal_count + [0]*palmar_count
    Tg = np.ones_like(obj)
    # order = np.append(order, np.array([len(Tg)-1, similarity_order[-1][0]]))
    if reverse:
        Tg = Tg.T
    order = order[:len(Tg)-1]
    n, _ = Tg.shape
    # print(n) # 100
    order_dict = dict()
    count = 0
    for row, label in zip(order, dorsals):
        # print(row[0], row[1])
        order_dict[row] = count
        count += 1
    # order_dict[]
    # order_dict[] = len(Tg)-1
    small_similarity_matrix = similarity_order[:, :k + 1]
    order_dict[similarity_order[-1][0]] = len(Tg) - 1
    # print(small_similarity_matrix[0]) #(100, 10 or K)
    for row in range(n):
        for sim in small_similarity_matrix[row, 1:]:
            # print(order_dict[sim])
            try:
                Tg[row][order_dict[sim]] = 1
            except:
                pass
    # Tg[order[-1]][]
    # print(Tg[8])
    Tg = Tg.T
    # The matrix Tg is a graph with 5 outgoing edges from each node, each node represented vertically
    Tg = Tg / Tg.sum(axis=0)
    # print(Tg[:,0])
    # order_dict has ordered index of each file, so given key filename, fetches index from the range of number of files

    # The restart or teleport matrix S is ready below
    S = np.zeros((len(Tg)))
    S[-1] = 1
    # print(S.shape) # (100,)
    # prev = np.zeros(Tg.shape)
    # print(Tg.shape)
    Tg[len(Tg)-1, :] = 1
    piv = np.linalg.inv(np.eye(len(Tg)) - (1 - p) * Tg) @ (p * S)
    # print(piv.shape)
    # row_sums = np.sum(piv, axis=1)
    tops = [x for _,x in sorted(zip(piv, dorsals), reverse=True)][:k]
    pred = scipy.stats.mode(tops)[0]
    confidence = tops.count(pred[0])/len(tops)
    return pred, confidence


def ppr(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids, unlabeled_folder=None, k=9):
    # dorsal_labels = np.array([[1]] * len(dorsal_image_ids))
    # palmar_labels = np.array([[0]] * len(palmar_image_ids))
    # dorsal_train = np.hstack((dorsal_data, dorsal_labels))
    # palmar_train = np.hstack((palmar_data, palmar_labels))
    dorsal_count = len(dorsal_image_ids)
    palmar_count = len(palmar_image_ids)
    # labels = []
    all_train = np.vstack((dorsal_data, palmar_data))
    all_image_ids = np.append(dorsal_image_ids, palmar_image_ids)
    predictions, confidence = [], []
    # img_order = [i,img_order[i] for i in range(len(img_order))]
    for id, ul_img in enumerate(ul_data):
        # print(all_train.shape, ul_img.shape)
        curr_test_matrix = np.vstack((all_train, ul_img))
        # print(all_train.shape, ul_img.shape)
        objs, similarity_order, img_order = find_similarity(curr_test_matrix, np.append(all_image_ids, np.array(ul_image_ids[id])))
        # img_order.append(ul_image_ids[id])
        pred, conf = pagerank(objs, similarity_order, np.append(img_order,ul_image_ids[id]), dorsal_count, palmar_count, k)
        predictions.append(pred)
        confidence.append(conf)
    if unlabeled_folder is None:
        return predictions, confidence
    ul_dorsal, _ = get_images_by_metadata(ul_image_ids, ul_data, unlabeled_folder, dorsal=1)
    ul_palmar, _ = get_images_by_metadata(ul_image_ids, ul_data, unlabeled_folder, dorsal=0)
    return predictions, confidence


def get_label(labeled_images_folder, unlabeled_images_folder, k):
    """
    Given a folder with unlabeled images, the system labels
them as dorsal or palmer
    :param folder: The folder containing images for training
    :param k: The number of latent dimensions to compute
    :return: the labels for the unlabeled images
    """
    # Extract features for the input folder
    features_data, object_ids = feature_extractor.get_data_matrix(labeled_images_folder)

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = feature_extractor.get_data_matrix(unlabeled_images_folder)

    # Get dorsal images
    dorsal_data, dorsal_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, palmar_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=0)

    unlabelled_data, unlabelled_image_ids = get_images_by_metadata(ul_object_ids, ul_features_data,
                                                                   unlabeled_images_folder)

    return dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, unlabelled_data, unlabelled_image_ids

