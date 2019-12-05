import math
import numpy as np

def convert_to_binary(data):
    diviser = data.ptp(0)
    diviser[diviser==0] = 1
    data_normed = np.divide((data - data.min(0)), diviser)
    for i in range(1, data.shape[1]):
        cov = np.cov(np.stack((data_normed[:,0], data_normed[:,i]), axis=0))
        if cov[0][1] < 0 :
            data_normed[:, i] = 1 - data_normed[:, i]
        data_normed[data_normed>=0.5] = 1
        data_normed[data_normed<0.5] = 0
    return data_normed

def get_data_order(data, data_ids, relevant=[], irrelevant=[]):
    sim_scores = []

    r_true = [0]*data.shape[1]
    r_false = [0]*data.shape[1]
    i_true = [0]*data.shape[1]
    i_false = [0]*data.shape[1]

    for data_id, item in zip(data_ids, data):
        if data_id in relevant:
            for index, val in enumerate(item):
                if val == 1:
                    r_true[index] += 1
                else:
                    r_false[index] += 1
        elif data_id in irrelevant:
            for index, val in enumerate(item):
                if val == 1:
                    i_true[index] += 1
                else:
                    i_false[index] += 1
    
    p = np.divide(
        np.add(r_true, 0.5),
        np.add(np.add(r_true, r_false), 1)
    )

    u = np.divide(
        np.add(i_true, 0.5),
        np.add(np.add(i_true, i_false), 1)
    )

    log_val = np.log(
        np.divide(
            np.multiply(p, np.subtract(1, u)),
            np.multiply(np.subtract(1, p), u)
        )
    )
    for data_id, item in zip(data_ids, data):
        similarity = np.sum(
            np.multiply(item, log_val)
        )
        sim_scores.append([data_id, similarity])
    
    sorted_ids = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_ids]