import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import feature_extractor

#def dtree(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids, unlabeled_folder):
def do_svm(features_data, object_ids, relevant_image_ids, irrelavent_image_ids, epochs = 1, alpha = 0.005):

    #Remove rows from data matrix for images which are not labelled
    #Do a copy instead of pointing to features_data
    x_train = features_data.copy()
    y_train = [0 for x in object_ids]
    for i in range(len(object_ids)):
        if object_ids[i] in relevant_image_ids:
            y_train[i] = 1
        elif object_ids[i] in irrelavent_image_ids:
            y_train[i] = -1
    length = (len(y_train))
    i, j = 0, 0
    while (j<length):
        if y_train[i] == 0:
            x_train = np.delete(x_train, i, 0)
            y_train.pop(i)
            i -= 1
        i += 1
        j += 1

    y_train = np.array(y_train)
    x_test = features_data.copy()

    #Training Begins
    w = np.zeros((x_train.shape[0],x_train.shape[1]))
    curr_time = time.time()
    while(epochs < 26):
        y = np.zeros((y_train.shape[0], 1))
        for i in range(x_train.shape[1]):
            y += w[:, i].reshape(x_train.shape[0],1) * x_train[:, i].reshape(x_train.shape[0],1)
        pred = y * y_train.reshape(y_train.shape[0],1)
        if (epochs % 5) == 0:
            #print(epochs, "epochs - %s seconds" % (time.time() - curr_time))
            curr_time = time.time()
        count = 0
        for val in pred:
            if(val >= 1):
                cost = 0
                for i in range(w.shape[1]):
                    w[:, i] -= alpha * (2 * 1/epochs * w[:, i])            
            else:
                cost = 1 - val
                for i in range(w.shape[1]):
                    w[:, i] += alpha * (x_train[count, i] * y_train[count] - 2 * 1/epochs * w[:, i])
            count += 1
        epochs += 1

    #What if the number of images are not equal in both folders? Reshape W!!
    w = np.array(list([w[0],] * x_test.shape[0]))
    
    #Make Predictions
    y_pred = np.zeros((x_test.shape[0], 1))
    for i in range(x_test.shape[1]):
        y_pred += w[:, i].reshape(x_test.shape[0],1) * x_test[:, i].reshape(x_test.shape[0],1)

    y_pred = list(y_pred.flatten())
    img_preds = [list(x) for x in zip(y_pred, object_ids)]
    sorted_img_preds = sorted(img_preds, reverse=True)

    relevance_list = [row[1] for row in sorted_img_preds]

    return(relevance_list)