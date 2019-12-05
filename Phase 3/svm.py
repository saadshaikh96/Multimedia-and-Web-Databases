import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import feature_extractor

def accuracy(list1, list2):
    length = len(list1)
    same_count = 0
    for i in range(length):
        if list1[i] == list2[i]:
            same_count += 1
    return(same_count*100/length)

def do_svm(dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, ul_data, ul_image_ids, max_epochs = 200, alpha = 0.01):
    #Use already fetched labelled data instead of fetching again.
    data_matrix = np.vstack((dorsal_data, palmar_data))
    object_ids = dorsal_image_ids + palmar_image_ids
    
    train_label_df = pd.read_excel("HandInfo.xlsx")
    test_label_df = pd.read_excel("HandInfo.xlsx")

    x_train = data_matrix
    x_test = ul_data
    w = np.zeros((x_train.shape[0],x_train.shape[1]))
    
    y_labels = []
    for i in range(len(object_ids)):
        if (list(train_label_df[train_label_df["imageName"] == object_ids[i]]["aspectOfHand"])[0][0] == 'd'):
            #Dorsal label is 1
            y_labels.append(1)
        elif (list(train_label_df[train_label_df["imageName"] == object_ids[i]]["aspectOfHand"])[0][0] == 'p'):
            #Palmar label is -1
            y_labels.append(-1)
        else:
            print("Label Read Error!!!")
            print(train_label_df[train_label_df["imageName"] == object_ids[i]]["aspectOfHand"])
    y_train = np.array(y_labels)

    y_labels = []
    for i in range(len(object_ids)):
        if (list(test_label_df[test_label_df["imageName"] == ul_image_ids[i]]["aspectOfHand"])[0][0] == 'd'):
            #Dorsal label is 1
            y_labels.append(1)
        elif (list(test_label_df[test_label_df["imageName"] == ul_image_ids[i]]["aspectOfHand"])[0][0] == 'p'):
            #Palmar label is -1
            y_labels.append(-1)
        else:
            print("Label Read Error!!!")
            print(test_label_df[test_label_df["imageName"] == ul_image_ids[i]]["aspectOfHand"])
    y_test = np.array(y_labels)

    #Training Begins
    w = np.zeros((x_train.shape[0],x_train.shape[1]))
    curr_time = time.time()
    epochs = 1
    while(epochs < max_epochs):
        y = np.zeros((y_train.shape[0], 1))
        for i in range(x_train.shape[1]):
            y += w[:, i].reshape(x_train.shape[0],1) * x_train[:, i].reshape(x_train.shape[0],1)
        pred = y * y_train.reshape(y_train.shape[0],1)
        if (epochs % 20) == 0:
            print(epochs, "epochs - %s seconds" % (time.time() - curr_time))
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
    predictions = []
    for val in y_pred:
        if(val > 0):
            predictions.append(1)
        else:
            predictions.append(-1)
    
    for i in range(len(predictions)):
        prediction_word = "Dorsal"
        if predictions[i] == -1:
            prediction_word = "Palmar"
        print("For unlabelled image -", ul_image_ids[i], "- Prediction is -", prediction_word)

    print("Accuracy is -", accuracy(y_test,predictions))

    return(predictions)