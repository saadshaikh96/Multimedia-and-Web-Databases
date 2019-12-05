import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

excel_sheet = "HandInfo.xlsx"

def get_images_by_metadata(object_ids, data, folder, male=-1, accessories=-1, dorsal=-1, left=-1):
    imagesToUse = []
    binary_matrix = get_binary_matrix(folder)
    for row in binary_matrix:
        if (male == -1 or row[1] == male) and (accessories == -1 or row[2] == accessories) and (
                dorsal == -1 or row[3] == dorsal) and (left == -1 or row[4] == left):
            imagesToUse.append(row[0])
    new_data = []
    new_object_ids = []
    for objName, objFeatures in zip(object_ids, data):
        if objName in imagesToUse:
            new_data.append(objFeatures)
            new_object_ids.append(objName)

    new_data = np.asarray(new_data)
    return new_data, new_object_ids


def get_binary_matrix(folder):
    imgpaths = os.listdir(folder)
    imagePaths = []
    for image in imgpaths:
        if len(image) < 1:
            continue
        if image[0] != "H":
            continue
        imagePaths.append(image.split('\\')[-1].split("/")[-1])
    df = pd.read_excel(excel_sheet)

    binary_matrix = []
    for i, imageName in enumerate(df['imageName']):
        if imageName in imagePaths:
            item = [imageName]
            if df.loc[i, :]["gender"].lower() == "male":
                item.append(1)
            else:
                item.append(0)
            item.append(int(df.loc[i, :]["accessories"]))
            lr_dp = df.loc[i, :]["aspectOfHand"].split(" ")
            if lr_dp[0] == "dorsal":
                item.append(1)
            else:
                item.append(0)
            if lr_dp[1] == "left":
                item.append(1)
            else:
                item.append(0)
            binary_matrix.append(item)
    return binary_matrix

def get_metadata_by_id(image_id, folder_name):
    pass


def displayResults(source_image, similar_images):
    plt.figure()
    num_figures = len(similar_images)+1
    num_columns = 5
    num_rows = int(math.ceil(num_figures/num_columns))
    plt.subplot(num_rows, num_columns, 1)
    s_img = mpimg.imread(source_image)
    plt.imshow(s_img)

    for sid, sim_image in enumerate(similar_images):
        plt.subplot(num_rows, num_columns, sid+2)
        img_matrix = mpimg.imread(sim_image)
        plt.imshow(img_matrix)

    plt.show()