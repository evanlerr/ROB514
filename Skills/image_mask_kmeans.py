#!/usr/bin/env python3

# This assignment introduces you to a common task (creating a segmentation mask) and a common tool (kmeans) for
#  doing clustering of data and the difference in color spaces.

# There are no shortage of kmeans implementations out there - using scipy's
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten

# Using imageio to read in the images and skimage to do the color conversion
import imageio
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt


def read_and_cluster_image(image_name, use_hsv, n_clusters):
    """ Read in the image, cluster the pixels by color (either rgb or hsv), then
    draw the clusters as an image mask, colored by both a random color and the center
    color of the cluster
    @image_name - name of image in Data
    @use_hsv - use hsv, y/n
    @n_clusters - number of clusters (up to 6)"""

    # Read in the file
    im_orig = imageio.imread("Skills/Data/" + image_name) ## MAY HAVE TO DELETE SKILSL LATER
    # Make sure you just have rgb (for those images with an alpha channel)
    im_orig = im_orig[:, :, 0:3]

    # The plot to put the images in
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Make name for the image from the input parameters
    str_im_name = image_name.split('.')[0] + " "
    if use_hsv:
        str_im_name += "HSV"
    else:
        str_im_name += "RGB"

    str_im_name += f", k={n_clusters}"

    # This is how you draw an image in a matplotlib figure
    axs[0].imshow(im_orig)
    # This sets the title
    axs[0].set_title(str_im_name)

    # TODO
    # Step 1: If use_hsv is true, convert the image to hsv (see skimage rgb2hsv - skimage has a ton of these
    #  conversion routines)

    if use_hsv:
        im_orig = rgb2hsv(im_orig)
    #else:
        #im_orig = rgb2hsv(im_orig) # just to get all values on the same scalar range haha
        #im_orig = hsv2rgb(im_orig)
    sz1 = np.size(im_orig,0)
    sz2 = np.size(im_orig,1)
    sz3 = np.size(im_orig,2)

    im_proc = np.reshape(im_orig,[sz1*sz2, sz3]).astype(float)
    sr1 = np.size(im_proc,0)
    sr2 = np.size(im_proc,1)

    means = np.zeros(3)
    stds = np.zeros(3)

    # whitening manually hahhahahahahahahah
    for i in range(np.size(im_proc,1)):
        means[i] = np.mean(im_proc[:,i])
        stds[i] = np.std(im_proc[:,i])
        im_proc[:,i] = (im_proc[:,i]-means[i])/stds[i]

    centers,_  = kmeans(im_proc,n_clusters)
    ids,_ = vq(im_proc,centers)

    rgb_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    #rgb_means = []
    imdata = im_proc.copy()
    masked_pix = np.array([rgb_color[int(i)] for i in ids])
    mask_image = masked_pix.reshape(sz1,sz2,3)
    axs[1].imshow(mask_image)


    for i in range(n_clusters):
        imdata[ids == i, 0:3] = centers[i]
    rgb_color = imdata * stds + means
    rgb_color = np.clip(np.reshape(rgb_color,[sz1,sz2,3]),0,255)


    rgb_means = np.zeros((n_clusters,3))
    for i in range(n_clusters):
        rgb_means[i] = np.mean(im_proc[ids == i],axis = 0)
    rgb_means = rgb_means * stds + means #unwhiten
    img_means = rgb_means[ids]
    if use_hsv:
        img_means = hsv2rgb(img_means)
    else:
        img_means = np.clip(img_means, 0, 255).astype(np.uint8)
    img_means = np.reshape(img_means,[sz1,sz2,sz3])
    
    axs[2].imshow(img_means)
    # Step 2: reshape the data to be an nx3 matrix
    #   kmeans assumes each row is a data point. So you have to give it a (widthXheight) X 3 matrix, not the image
    #   data as-is (WXHX3). See numpy reshape.
    # Step 3: Whiten the data
    # Step 4: Call kmeans with the whitened data to get out the centers
    #   Note: kmeans returns a tuple with the centers in the first part and the overall fit in the second
    # Step 5: Get the ids out using vq
    #   This also returns a tuple; the ids for each pixel are in the first part
    #   You might find the syntax data[ids == i, 0:3] = rgb_color[i] useful - this gets all the data elements
    #     with ids with value i and sets them to the color in rgb_color
    # Step 5: Create a mask image, and set the colors by rgb_color[ id for pixel ]
    # Step 6: Create a second mask image, setting the color to be the average color of the cluster
    #    Two ways to do this
    #       1) "undo" the whitening step on the returned cluster (harder)
    #       2) Calculate the means of the clusters in the original data
    #           np.mean(data[ids == c])
    # Note: To do the HSV option, get the RGB version to work. Then go back and do the HSV one
    #   Simplest way to do this: Copy the code you did before and re-do after converting to hsv first
    #     Don't forget to take the color centers in the *original* image, not the hsv one
    #     Don't forget to rename your variables
    #   More complicated: Make a function. Most of the code is the same, except for a conversion to hsv at the beginning
    print("hello world")
    # An array of some default color values to use for making the rgb mask image
    rgb_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    # YOUR CODE HERE
    axs[1].set_title("ID colored by rgb")
    axs[2].set_title("ID colored by cluster average")


if __name__ == '__main__':
    read_and_cluster_image("real_apple.jpg", True, 4)
    read_and_cluster_image("trees.png", True, 2)
    read_and_cluster_image("trees_depth.png", False, 3)
    read_and_cluster_image("staged_apple.png", True, 3)
    read_and_cluster_image("staged_apple.png", False, 3)
    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    plt.show()
    print("done")

# List of names (creates a set)
worked_with_names = {"mindy atuschul"}
# List of URLS FW25(creates a set)
websites = {"scipy tutorials","https://docs.scipy.org/doc/scipy/reference/cluster.vq.html"}
# Approximate number of hours, including lab/in-class time
hours = 7