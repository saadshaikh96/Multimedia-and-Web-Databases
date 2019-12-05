import os
import cv2
import json
from skimage.feature import hog
from scipy.stats import skew as scipy_skew
from skimage.transform import rescale, downscale_local_mean
from tqdm import tqdm

import numpy as np

def get_hog_features_by_image_path(path):
    # Given image path generate HOG features
    image = cv2.imread(path)
    image = rescale(image, 0.1)
    # BGR to RGB
    image = image[:, :, ::-1]
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd = np.ravel(fd).astype('float32')
    return fd.tolist()

def get_cm_features_by_image_path(path):
    # Given path generate color moments
    image = cv2.imread(path)
    # BGR to RGB
    image = image.astype('float32')
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    channels = cv2.split(img_yuv)  # y, u, v
    total_vector = []
    for channel in channels:
        channel_vector = []
        for i in range(0, channel.shape[0], 100):
            for j in range(0, channel.shape[1], 100):
                image_segment = channel[i:i + 100, j:j + 100]
                mean = image_segment.mean()
                var = image_segment.var()
                skew = scipy_skew(image_segment, axis=None)
                channel_vector += [mean, var, skew]
        total_vector += channel_vector
    return total_vector

def get_files_in_folder(folder):
    # Get list of files given a folder path
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def get_data_matrix(folder, method="hog"):
    features = []
    object_ids = []
    files = get_files_in_folder(folder)
    count = 0
    for file in tqdm(files):
        if count % 100 == 0:
            print("processed %s files" % count)
        count += 1
        if method == "hog":
            features_one = get_hog_features_by_image_path(file)
        elif method == "cm":
            features_one = get_cm_features_by_image_path(file)
        object_ids.append(os.path.basename(file))
        features.append(features_one)
    data_matrix = np.asarray(features)
    print(data_matrix.shape)
    return data_matrix, object_ids

def load_saved_features(folder, method="hog"):
    data_dict = {}
    with open("%s.json" % method, "r") as f:
        data_dict = json.loads(f.read())
    return np.asarray(data_dict['data']), data_dict['ids']

def extract_and_save_features(folder, method="hog"):
    data_matrix, object_ids = get_data_matrix(folder, method)
    data_dict = {
        'data': data_matrix.tolist(),
        'ids': object_ids
    }
    with open("%s.json" % method, "w") as f:
        f.write(json.dumps(data_dict))
