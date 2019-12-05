# python task3.py -k 5 -K 10 -f ./phase3_sample_data/Labelled/Set2 -i 'Hand_0008333.jpg' 'Hand_0006183.jpg' 'Hand_0000074.jpg'

import argparse
import cv2
import os
import skimage
from skimage.feature import local_binary_pattern
import matplotlib.image as mpimg
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Find most similar K images for Personalized Page Rank given 3 restarts')
parser.add_argument('-k', type=int, default=5, help='The number of nearest neighbors')
parser.add_argument('-K', type=int, default=10, help='The number of popular images to be visualized')
parser.add_argument('-f', required=True, type=str, default='Labelled/Set2', help='the folder where images are stored')
parser.add_argument('-i', required=True, nargs=3, type=str, help='the image IDs')

vectors_file = 'rank vectors initial.csv'
order_file = 'order.csv'
similarity_order_file = 'similarity_order.csv'


def sort_list(list1, list2):
    '''
    :param list1: the list that is to be sorted
    :param list2: the list that is to provide the sorting order
    :return: a list of elements in list1 sorted reversed by the values in list2
    '''
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    # print(z)
    y = [x for x in reversed(z)]
    return y


def get_similar_images_for_each(lbp_outputs):
    objs = np.ones((len(lbp_outputs), len(lbp_outputs)))
    for i in range(len(lbp_outputs)):
        i_vec = np.array(lbp_outputs[i][1])
        i_mag = np.linalg.norm(i_vec)
        for j in range(i + 1, len(lbp_outputs)):
            j_vec = np.array(lbp_outputs[j][1])
            j_mag = np.linalg.norm(j_vec)
            objs[i, j] = objs[j, i] = (i_vec @ j_vec.T) / (i_mag * j_mag)
    return objs


def get_lbp_features_by_image_path(folder, file):
    # print(os.path.join(folder, file))
    im = cv2.imread(os.path.join(folder, file))
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp_output = []
    radius = 1.0
    number_of_points = 8 * radius
    eps = 1e-6
    # Computing LBP values for each (100x100) window
    for i in range(12):
        block_lbp = []
        for j in range(16):
            lbp_window = skimage.feature.local_binary_pattern(im_gray[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100],
                                                              number_of_points, radius, method='uniform')
            lbp_window_histogram, _ = np.histogram(lbp_window.flatten(), bins=np.arange(0, number_of_points + 3),
                                                   range=(0, number_of_points + 2))
            lbp_window_histogram = lbp_window_histogram.astype('float')
            lbp_window_histogram /= (lbp_window_histogram.sum() + eps)
            block_lbp.append(lbp_window_histogram)
        # Storing LBP values as (12x160) array
        lbp_output.append(list((round(item, 4) for sublist in block_lbp for item in sublist)))
    lbp_feature_vector = list((item for sublist in lbp_output for item in sublist))
    return lbp_feature_vector


def plot_images(original_image_path, similar_images):
    # Plot similar images for task 3
    # Each item in similar_image is [<image_path>, <similarity>]
    total_plots = len(similar_images) + 1
    cols = rows = math.ceil(math.sqrt(total_plots))
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    index = 0
    for row in axes:
        for ax in row:
            if index < len(similar_images):
                similar_img = similar_images[index]
                # print(similar_img)
                if similar_img[0][-1] == '\'':
                    filename = similar_img[0].split('\'')[1]
                else:
                    filename = similar_img[0]
                # print(filename)
                img = mpimg.imread(os.path.join(original_image_path, filename))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(filename + "(" + str(round(similar_img[1], 6)) + ")")
                index += 1
            else:
                ax.axis('off')
    # plt.show()


def visualize(x, K, folder, original_images):
    order = np.array(pd.read_csv(order_file).as_matrix())
    # print(order)
    order = order[:, 1]
    # print(order)
    originals = [[id, 1] for id in original_images]
    plot_images(folder, originals)
    # get the x vector sorted reverse by the order in order
    similar_images = sort_list(order, x)
    scores = sorted(x, reverse=True)
    high_scorers = [[img_id, score] for img_id, score in zip(similar_images[:K], scores)]
    plot_images(folder, high_scorers)
    plt.show()


def pagerank(k=5, p=0.85, personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph

    Inputs:
    -------
    k: number of connections from each node
    p: damping factor
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically.
    reverse: If true, returns the reversed-PageRank

    Returns:
    --------
    PageRank Scores for the nodes

    """

    order = np.array(pd.read_csv(order_file).as_matrix())
    # print(order.shape) # (100, 2)
    obj = genfromtxt(vectors_file, delimiter=',')[1:]
    similarity_order = np.array(pd.read_csv(similarity_order_file).as_matrix())
    obj = obj[:, 1:]
    Tg = np.zeros_like(obj)
    # print(Tg.shape) # (100, 100)

    if reverse:
        Tg = Tg.T

    n, _ = Tg.shape
    # print(n) # 100
    order_dict = dict()
    for row in order:
        order_dict[row[1]] = row[0]

    small_similarity_matrix = similarity_order[:, :k + 1]
    # print(small_similarity_matrix[0]) #(100, 10 or K)
    for row in range(n):
        for sim in small_similarity_matrix[row, 1:]:
            # print(order_dict[sim])
            Tg[row][order_dict[sim]] = 1
    # print(Tg[8])
    Tg = Tg.T
    # The matrix Tg is a graph with 5 outgoing edges from each node, each node represented vertically
    Tg = Tg / Tg.sum(axis=0)
    # print(Tg[:,0])
    # order_dict has ordered index of each file, so given key filename, fetches index from the range of number of files

    # The restart or teleport matrix S is ready below
    S = np.zeros((len(order_dict)))
    for key in personalize:
        S[order_dict[key[1:-1]]] = (1 - p) / len(personalize)
    # print(S.shape) # (100,)
    # prev = np.zeros(Tg.shape)
    # print(Tg.shape)

    piv = np.linalg.inv(np.eye(100) - (1 - p) * Tg) @ (p * S)
    print(piv)
    return piv


def find_similarity(folder):
    """
    Given a folder with images, the pageranks are visualized
    :param folder: Folder containing the images
    :return: saves the three files with order of rows, similarity matrix, most similar images for each image
    """
    # TODO  write your code here
    lbp_outputs = []
    img_order = []
    for pic in os.listdir(folder):
        img_order.append(pic)
        temp = np.array(get_lbp_features_by_image_path(folder, pic))
        temp = temp.ravel()
        lbp_outputs.append([pic, temp])
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
    pd.DataFrame(obj).to_csv(vectors_file)
    pd.DataFrame(similarity_order).to_csv(similarity_order_file)
    pd.DataFrame(img_order).to_csv(order_file)


if __name__ == "__main__":
    args = parser.parse_args()
    k = args.k
    K = args.K
    target_folder = args.f
    image_ids = args.i
    RESTART_FACTOR = 0.85

    # comment the below line to continue using the saved vectors, uncomment to calculate again
    find_similarity(target_folder)
    # gets the nxn matrix and perform topic based pagerank for given parameters
    x = pagerank(k, p=RESTART_FACTOR, personalize=image_ids)  # , max_iter = 100
    # print(x.shape) # (100,100)
    visualize(x, K, target_folder, image_ids)
