import os
import math
import json
import numpy as np

import argparse
from feature_extractor import get_data_matrix, extract_and_save_features, load_saved_features
from pca import get_pca_decomposition, get_pca_transform
from svd import get_svd_decomposition, get_svd_transform
from utils import get_images_by_metadata
from kmeans import get_cluster_centers
from kmeans2 import get_centroids
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from task5 import get_similar

import feedback_prob
import feedback_dt
import feedback_ppr
import svm_t6

parser = argparse.ArgumentParser(description='Relevant feedback system')
parser.add_argument('-s', type=str, default="prob", help='Feedback system prob/svm/dt/ppr')

def refine_results(feedback_system):
    relavent = []
    irrelavent = []
    i = 0

    # Get 1000 similar images from task 5b
    if os.path.exists('t5_output.json'):
        with open('t5_output.json', 'r') as cache_file:
            data_cache = json.load(cache_file)
            features_data = np.array(data_cache['data'])
            object_ids = data_cache['ids']
    else:
        with open('cm.json', 'r') as cache_file:
            lsh_cache = json.load(cache_file)
            features = np.array(lsh_cache['data'])
            filepaths = lsh_cache['ids']

        features_data, object_ids = get_similar(
            10,
            10,
            features,
            filepaths,
            os.path.join("Hands", "Hand_0000674.jpg"),
            1000)

        with open('t5_output.json', 'w') as cache_file:
            tmp_cache = {}
            tmp_cache['data'] = features_data
            tmp_cache['ids'] = object_ids
            json.dump(tmp_cache, cache_file)

    print ("Similar images ready")

    features_data = np.asarray(features_data)

    if feedback_system == "prob":
        features_data = feedback_prob.convert_to_binary(features_data)

    print ("binary data converted")

    id_mapper = {}
    id_reverse_map = {}
    for index, item in enumerate(object_ids):
        id_mapper[index] = item
        id_reverse_map[item] = index

    sorted_ids = object_ids

    print ("starting algo")

    while(1):        
        plot_images(sorted_ids[:20], "Hands", id_reverse_map)

        print("Enter Relavent images (eg. 1 2 4)")
        relavent = np.hstack((relavent, [int(x) for x in (input()).split(" ")]))
        print("Enter Irrelavent images (eg. 3 5 6 8)")
        irrelavent = np.hstack((irrelavent, [int(x) for x in (input()).split(" ")]))
        relavent = list(set(relavent))
        irrelavent = list(set(irrelavent))

        for item in set(relavent).intersection(set(irrelavent)):
            print("Removing %s because it was marked as both relavent and irrelavent." % item)
            relavent.remove(item)
            irrelavent.remove(item)

        relevant_image_ids = [id_mapper[x] for x in relavent]
        irrelavent_image_ids = [id_mapper[x] for x in irrelavent]

        if feedback_system == "prob":
            sorted_ids = feedback_prob.get_data_order(features_data, object_ids, relevant_image_ids, irrelavent_image_ids)
        elif feedback_system == "svm":
            sorted_ids = svm_t6.do_svm(features_data, object_ids, relevant_image_ids, irrelavent_image_ids)
        elif feedback_system == "dt":
            sorted_ids = feedback_dt.get_data_order(features_data, object_ids, relevant_image_ids, irrelavent_image_ids)
        elif feedback_system == "ppr":
            sorted_ids = feedback_ppr.get_data_order(features_data, object_ids, relevant_image_ids, irrelavent_image_ids)
        i += 1
        print(relavent)

def plot_images(images, folder_path, mapper):
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
                ax.set_title(filename + "(%s)" % mapper[filename])
                index += 1
            else:
                ax.axis('off')
    plt.show(block=False)

if __name__ == "__main__":
    args = parser.parse_args()
    feedback_system = args.s
    refine_results(feedback_system)