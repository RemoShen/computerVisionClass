import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def get_distances(data, point):
    """ Calculate the distance between the data set and a specific point """
    return cdist(data.T, point.reshape((-1, 1)).T)[:, 0]


def calc_mean_and_basin(data, point, r, c):
    """
    Calculate the mean point in the circle with center "point" and radius "r"
    :param data: n-dimensional dataset containing p points
    :param point: center of the circle of search
    :param r: radius of the circle
    :param c: radius of the circle for the basin of attraction
    :return: mean point, basin of attraction
    """
    distances = get_distances(data, point)
    mean_point = np.mean(data[:, distances < r], axis=1)

    distances_to_mean = get_distances(data, mean_point)
    basin_of_attraction = np.argwhere(distances_to_mean < r / c)[:, 0]

    return mean_point, basin_of_attraction


def plot_clusters_3d(data, labels, peaks, title):
    """
    Plots the modes of the given image data in 3D by coloring each pixel
    according to its corresponding peak.
    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as RGB values.
        title: title for the plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for idx, peak in enumerate(peaks):
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[peak], s=.5, label="Group =" + str(idx))

    plt.title(title)
    # plt.legend(markerscale=10)
    # plt.savefig("../result_imgs/cluster_" + title.replace(" = ", "-").replace(". ", "_"))
    plt.show()


def plot_dict(dictionary):
    cols = len(dictionary)
    i = 0
    plt.figure()
    for title, image in dictionary.items():
        i += 1
        plt.subplot(1, cols, i)
        plt.imshow(image)
        plt.title(title)
    plt.show()