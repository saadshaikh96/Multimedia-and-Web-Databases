#! /usr/bin/env python3
import argparse
import pickle
import os
import json
from feature_extractor import get_cm_features_by_image_path, get_data_matrix, get_files_in_folder
from lsh import LSH
import numpy as np
from utils import displayResults
import base64

parser = argparse.ArgumentParser(description='Create LSH in memory index. Get t most similar images')
parser.add_argument('-l', required=True, type=int, default=5, help='Number of Layers')
parser.add_argument('-k', required=True, type=int, help='Number of hashes per layer')
parser.add_argument('-i', required=True, type=str, help='Path of the image DB(folder)')
parser.add_argument('-q', type=str, help='Query image path')
parser.add_argument('-t', type=int, help='Number of similar images')

def get_similar(num_layers, num_hashes, features, filepaths, query_image, num_results):
    lsh = LSH(num_layers, num_hashes, 10)
    lsh.fit(features)

    query_vec = get_cm_features_by_image_path(query_image)
    similar_indices = lsh.get_similar(query_vec, num_results)[:num_results]

    return [list(features[x]) for x in similar_indices], [filepaths[x] for x in similar_indices]


if __name__ == "__main__":
    args = parser.parse_args()
    num_layers = args.l
    num_hashes = args.k
    input_vectors = args.i
    query_image = args.q
    num_similar = args.t

    lsh_cache = {}
    filepaths = []
    if os.path.exists('cm.json'):
        with open('cm.json', 'r') as cache_file:
            lsh_cache = json.load(cache_file)
            features = np.array(lsh_cache['data'])
            filepaths = lsh_cache['ids']
    # if input_vectors in lsh_cache.keys():
    #     features = np.array(lsh_cache['data'])
    #     filepaths = lsh_cache['ids']
    else:
        if os.path.isfile(input_vectors):
            with open(input_vectors) as features_file:
                features = np.array(json.load(features_file))
        else:
            features, filenames = get_data_matrix(input_vectors, method='cm')
            filepaths = [os.path.join(input_vectors, x) for x in filenames]

        with open('cm.json', 'w') as cache_file:
            tmp_cache = {}
            tmp_cache['data'] = features.tolist()
            tmp_cache['ids'] = filepaths
            json.dump(tmp_cache, cache_file)


    fts, filenames = get_similar(num_layers, num_hashes, features, filepaths, query_image, num_similar)

    elevenk_folder = 'Hands'

    print(filenames)

    displayResults(query_image, [os.path.join(elevenk_folder, x) for x in filenames])
