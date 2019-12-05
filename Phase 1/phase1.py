import numpy as np
import pandas as pd
import scipy, time, cv2, os
import skimage, math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import feature
from tabulate import tabulate


'''Calculating color moments of the input image.
Returns a feature vector of color moments length 1728'''
def Color_Moments(im):

    # Converting  image to YUV color model
    im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    M1, M2, M3, cm_temp_vector, cm_feature_vector = [],[],[],[],[]
    # Calculating moments for each channel
    for i in range(12):
        for j in range (16):
            m1 = np.mean(im_yuv[i*100:(i+1)*100, j*100:(j+1)*100], axis=(0,1)).tolist()
            m2 = np.std(im_yuv[i*100:(i+1)*100, j*100:(j+1)*100], axis=(0,1)).tolist()
            reshaped_matrix_for_skew = np.reshape(im_yuv[i*100:(i+1)*100, j*100:(j+1)*100], (100*100,3))
            m3 = scipy.stats.skew(reshaped_matrix_for_skew).tolist()
            M1.append(m1), M2.append(m2), M3.append(m3)

    #Storing each moment as a list => len(cm_temp_vector) = 3
    cm_temp_vector.append(list((item for sublist in M1 for item in sublist)))
    cm_temp_vector.append(list((item for sublist in M2 for item in sublist)))
    cm_temp_vector.append(list((item for sublist in M3 for item in sublist)))
    # Creating Feature vector of the color moments => 1728 values (12x16x3x3)
    cm_feature_vector = list((round(item,4) for sublist in cm_temp_vector for item in sublist))

    return cm_feature_vector


# Calculating Local Binary Pattern of the input image.
def LocalBinaryPattern(im):

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Initializing LBP parameters
    lbp_output = []
    radius = 1.0
    number_of_points = 8 * radius
    eps = 1e-7
    # Computing LBP values for each (100x100) window
    for i in range(12):
        block_lbp = []
        for j in range (16):
            lbp_window = skimage.feature.local_binary_pattern(im_gray[i*100:(i+1)*100, j*100:(j+1)*100],\
                                                                number_of_points,radius, method='uniform')
            lbp_window_histogram, _ = np.histogram(lbp_window.flatten(), bins = np.arange(0, number_of_points+3),\
                                                    range = (0, number_of_points+2))
            lbp_window_histogram = lbp_window_histogram.astype('float')
            lbp_window_histogram /= (lbp_window_histogram.sum() + eps)
            block_lbp.append(lbp_window_histogram)
        # Storing LBP values as (12x160) array
        lbp_output.append(list((round(item,4) for sublist in block_lbp for item in sublist)))
    lbp_feature_vector = list((item for sublist in lbp_output for item in sublist))

    return lbp_feature_vector


def Euclidean(a, b):
    return (sum(pow(x-y,2) for x, y in zip(a, b))**0.5)

def pprint_df(dframe):
    print (tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))


def task_1(path, image):
    # Reading image from folder
    im = cv2.imread(os.path.join(path,image), 1)
    cm_feature_vector = Color_Moments(im)
    lbp_feature_vector = LocalBinaryPattern(im)
    input_imageFeatures_df = pd.DataFrame(columns=['Image', 'Color_moments_vector', 'LBP_vector'])

    input_imageFeatures_df = input_imageFeatures_df.append({'Image':image, \
                                                            'Color_moments_vector':cm_feature_vector, \
                                                            'LBP_vector': lbp_feature_vector},\
                                                            ignore_index=True)

    return input_imageFeatures_df


def task_2(path, picklepath):

    count = 0
    imageFeatures_df = pd.DataFrame(columns=['Image', 'Color_moments_vector', 'LBP_vector'])
    print (time.ctime())

    for image in os.listdir(path):
        input_imageFeatures_df = task_1(path, image)
        imageFeatures_df = imageFeatures_df.append(input_imageFeatures_df)
        count += 1
        if (count%1000 == 0):
            print ("Done: ", count, "   ", time.ctime())

    print ("Pickling now...", time.ctime())
    imageFeatures_df.to_pickle(picklepath)
    print ("Done Pickling:  ", time.ctime())


def task_3(path, picklepath):

        model = input("Model: ")
        imageID = input("Image ID: ")
        k = int(input("K: "))
        imageDescriptor_df = pd.read_pickle(picklepath)
        input_imageFeatures_df = task_1(path, imageID)
        similarity_df = pd.DataFrame(columns=['Image', 'Similarity_Score'])
        print("Calculating Similarity Scores: ")
        for i in range(len(imageDescriptor_df)):

            if model == "CM":
                distance = Euclidean(input_imageFeatures_df.iloc[0]['Color_moments_vector'],\
                                    imageDescriptor_df.iloc[i]['Color_moments_vector'])
            if model == "LBP":
                distance = Euclidean(input_imageFeatures_df.iloc[0]['LBP_vector'],\
                                    imageDescriptor_df.iloc[i]['LBP_vector'])

            similarity_df = similarity_df.append({'Image':imageDescriptor_df.iloc[i]['Image'], \
                                                    'Similarity_Score':distance}, ignore_index=True)


        similarity_df = similarity_df.sort_values('Similarity_Score')
        print ("\n\n\n Top ", k, " similar images are: \n\n")
        pprint_df(similarity_df.iloc[0:k+1])
        print ("\n\n\n")
        return imageID, similarity_df.iloc[0:k+1], model



def plot_images(original_image_path, similar_images):
	# Plot similar images for task 3
	# Each item in similar_image is [<image_path>, <similarity>]
	total_plots = len(similar_images) + 1
	cols = rows = math.ceil(math.sqrt(total_plots))
	fig, axes = plt.subplots(nrows=rows, ncols=cols)
	original_plotted = False
	index = 0
	for row in axes:
		for ax in row:
			if index < len(similar_images):
				if not original_plotted:
					img = mpimg.imread(original_image_path)
					ax.imshow(img)
					ax.axis('off')
					ax.set_title("Original Image")
					original_plotted = True
				else:
					similar_img = similar_images[index]
					filename = similar_img[0].split('\\')[-1]
					img = mpimg.imread(os.path.join(path,similar_img[0]))
					ax.imshow(img)
					ax.axis('off')
					ax.set_title(filename + "(" + str(round(similar_img[1], 2)) + ")")
					index += 1
			else:
				ax.axis('off')
	plt.show()

# *** Change value of dir to point to the location of the folder on your machine ***
dir = "/Users/saad/Desktop/ASU/Fall 2019/CSE 515 Multimedia and Web Databases/Phase1"

path = os.path.join(dir, "Test/testset1")
picklepath = os.path.join(dir, "Output/Task2.pkl")

task = int(input("Task: "))

if task == 1:
    image = input("ImageID: ")
    model = input("Model: ")
    input_imageFeatures_df = task_1(path, image)
    if model == "CM":
        input_imageFeatures_df.to_csv(os.path.join(dir, "Output/Task1_CM"), \
                                    columns=('Image', 'Color_moments_vector'), \
                                    index=False)
    if model == "LBP":
        input_imageFeatures_df.to_csv(os.path.join(dir, "Output/Task1_LBP"),\
                                    columns=('Image', 'LBP_vector'), index=False)
    print ("Created CSV")

if task == 2:
    task_2(path, picklepath)

if task == 3:
    inputimg,similarity_df, model = task_3(path, picklepath)
    if model == "CM":
        similarity_df.to_csv(os.path.join(dir, "Output/Task3_CM_Similarity"), index=False)
    if model == "LBP":
        similarity_df.to_csv(os.path.join(dir, "Output/Task3_LBP_Similarity"), index=False)

    similar_images = []
    for i in range(len(similarity_df['Image'])):
        display_image = similarity_df.iloc[i]['Image']
        score = similarity_df.iloc[i]['Similarity_Score']
        similar_images.append([display_image, score])
        # cv2.imshow(image, display_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    plot_images(os.path.join(path,inputimg), similar_images)
