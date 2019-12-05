import math
import numpy as np

from task4 import dtree

def get_data_order(data, data_ids, relevant=[], irrelevant=[]):

    relevant_data = [data[data_ids.index(i)] for i in relevant]
    relevant_data = np.asarray(relevant_data)
    relevant_ids = relevant

    irrelevant_data = [data[data_ids.index(i)] for i in irrelevant]
    irrelevant_data = np.asarray(irrelevant_data)
    irrelevant_ids = irrelevant

    preds = dtree(
        relevant_data,
        relevant_ids,
        irrelevant_data,
        irrelevant_ids,
        data,
        data_ids)

    relevant_dists = []
    irrelevant_dists = []
    for image_id, image_data, pred in zip(data_ids, data, preds):
        min_dist = None
        if pred == 1:
            for item in relevant_data:
                dist = euclidean(item, image_data)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            relevant_dists.append([image_id, min_dist])
        else:
            for item in irrelevant_data:
                dist = euclidean(item, image_data)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            irrelevant_dists.append([image_id, min_dist])

    relevant_dists = sorted(relevant_dists, key = lambda x: x[1])
    irrelevant_dists = sorted(irrelevant_dists, key = lambda x: x[1], reverse=True)
    
    return [x[0] for x in relevant_dists] + [x[0] for x in irrelevant_dists]

def euclidean(x,y):
    return np.linalg.norm(x-y)