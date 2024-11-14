import time

import scipy.io
from scipy.spatial.distance import pdist
from skimage.color import lab2rgb, rgb2lab
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from matplotlib import pyplot as plt

from helpers import *

iterations = 0


def plot_data(data_by_types):
    """ Show a scatter plot of the data by type, the first dimension groups per type """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for group, points in enumerate(data_by_types):
        ax.scatter(points[0], points[1], points[2], label="Group " + str(group))

    plt.legend()
    plt.show()


def find_peak_opt(data, idx, r, c, speed_up):
    """
    Define a spherical window at the data point of radius r and computing the mean of the points that lie within the
    window. Then shifts the window to the mean and repeats until convergence (the shift is under some threshold t).

    Speed-up techniques are:
        1. When the pick is found, all the points in the radius are also labeled
        2. In the search, all the points in a distance smaller or equal to r/c are also labeled

    :param data: n-dimensional dataset containing p points
    :param idx: index of the point we wish to calculate its associated density peak
    :param r: search window radius
    :param c: window radius for basin of attraction
    :param speed_up: flag to use speed-up techniques
    :return: mean point, ids of the point inside
    """

    point = data[:, idx]
    mean_point, basin_of_attraction = calc_mean_and_basin(data, point, r, c)

    if speed_up:
        # Creating a set ensures that the values inside appear only once
        ids_in_region = set(basin_of_attraction)
    else:
        ids_in_region = False

    threshold = 0.01

    global iterations

    while pdist(np.array([point, mean_point])) > threshold:
        iterations += 1
        point = mean_point
        mean_point, basin_of_attraction = calc_mean_and_basin(data, point, r, 4)
        if speed_up:
            ids_in_region.update(basin_of_attraction)

    if speed_up:
        _, basin_of_attraction = calc_mean_and_basin(data, point, r, 1)
        ids_in_region.update(basin_of_attraction)

    return np.array(mean_point), ids_in_region


def mean_shift_opt(data, r, c, speed_up):
    """
    Calculate mean shift
    :param data: n-dimensional dataset containing p points
    :param r: search window radius
    :param c: window radius for basin of attraction
    :param speed_up: flag to use speed-up techniques
    :return: labels, peaks, data grouped in labels
    """

    n_pixels = data.shape[1]

    # The label -1 means that is has no class
    labels = np.zeros(n_pixels) - np.ones(n_pixels)

    peaks = []
    processed_pixels = 0
    for i in tqdm(range(n_pixels)):
        if labels[i] != -1:
            continue

        processed_pixels += 1
        peak, ids_in_region = find_peak_opt(data, i, r, c, speed_up)
        if len(peaks) > 0:
            similarity = cdist(np.array(peaks), np.array([peak]))
            similar_peak = np.argwhere(similarity < r / 2)

            if len(similar_peak) > 0:
                labels[i] = similar_peak[0, 0]
                if speed_up:
                    labels[list(ids_in_region)] = similar_peak[0, 0]
                # print("i", i, "total", n_pixels)
                continue
            else:
                print("New peak", peak)
                labels[i] = int(len(peaks))
                if speed_up:
                    labels[list(ids_in_region)] = int(len(peaks))
                peaks.append(peak)
        else:
            print("First peak", peak)
            peaks.append(peak)
            labels[i] = 0
            if speed_up:
                labels[list(ids_in_region)] = 0

    # data_grouped = None
    used = round(100 * processed_pixels / n_pixels, 2)
    print("Total number of pixels:", n_pixels, ". Finished after", processed_pixels, "pixels processed ->",
          round(100 * processed_pixels / n_pixels, 2), "%")

    if np.min(labels) == -1:
        raise Exception("A pixel has not been assigned a group")

    return labels, np.array(peaks), [str(n_pixels), str(used)]


def debug_algorithm():
    data = scipy.io.loadmat('../pts.mat')['data']
    labels, peaks, _ = mean_shift_opt(data, 2, 4, False)
    data_grouped = [data[:, labels == group] for group in range(len(peaks))]
    plot_data(data_grouped)


def im_segmentation(im, r, c, make_5d, speed_up, title):
    """
    Segmentation applied to an image using mean-shift
    :param im: RGB of the image
    :param r: radius for the segmentation
    :param c: window radius for basin of attraction
    :param make_5d: flag to add position to the data for segmentation
    :param speed_up: flag to use speed_up techniques
    :param title: title for the resulting image
    :return: labels, peaks, data grouped
    """

    colors = rgb2lab(im)

    if make_5d:
        colors_5d = np.zeros((colors.shape[0], colors.shape[1], 5))
        for y in range(colors.shape[0]):
            for x in range(colors.shape[1]):
                colors_5d[y, x] = np.append(colors[y, x], [y, x])
        colors = colors_5d

    colors_reshaped = colors.reshape((colors.shape[0] * colors.shape[1], colors.shape[2])).T

    labels, peaks, performance = mean_shift_opt(colors_reshaped, r, c, speed_up)

    for i, peak in enumerate(peaks):
        peaks[i] = [int(value) for value in peak]

    colors_reshaped = colors_reshaped.T
    for i, pixel in enumerate(colors_reshaped):
        colors_reshaped[i] = peaks[int(labels[i])]

    segmented = lab2rgb(colors_reshaped.reshape(colors.shape)[:, :, :3])
    peaks = lab2rgb([peaks[:, :3]])[0, :, :]

    title += ". Peaks = " + str(len(peaks))

    plt.imshow(segmented)
    plt.title(title)
    plt.savefig("../result_imgs/" + title.replace(" = ", "-").replace(". ", "_"))
    plt.show()

    return labels, peaks, performance, segmented


def study_image(img_path, r, c, make_5d, opt, plot_3d=True, save_stats=True):
    """
    Study an image applying the mean shift algorithm
    :param img_path: string path of the image
    :param r: radius for the mean shift
    :param c: constant for the speed-up technique
    :param make_5d: Flag to use 5 dimensions or 3
    :param opt: Flag to use the speed-up techniques
    :param plot_3d: Flag to plot the 3D cluster
    :param save_stats: Flag to save the statistics
    :return: segmented image in rgb
    """
    global iterations
    iterations = 0
    start = time.time()

    # Load and show image
    img_rgb = plt.imread(img_path)
    # plt.imshow(img_rgb)
    # plt.show()

    title = "radius = " + str(r) + ". c = " + str(c) \
            + ". Dim = " + ("5D" if make_5d else "3D") \
            + ". Opt = " + ("Yes" if opt else "No")

    labels, peaks, performance, segmented = im_segmentation(img_rgb.copy(), r, c, make_5d, opt, title)

    title += ". Peaks = " + str(len(peaks))

    if plot_3d:
        plot_clusters_3d(img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], img_rgb.shape[2])), labels, peaks,
                         title)

    end = time.time()
    minutes = int((end - start) / 60)

    text = "Image " + img_path + " finished processing." + title + \
           ". Took " + str(minutes) + "m " + str(round(end - start - 60 * minutes)) + "s" \
           + ". Pixels = " + performance[0] + ", processed pixels = " + performance[1] + \
           "%. Iterations = " + str(iterations)
    if save_stats:
        with open("Output.txt", "a") as text_file:
            print(f'{text}', file=text_file)

    return segmented


def compare_dimensions(img_path, r):
    """
    Segment the same image with the same radius and plot them to see the differences
    :param img_path: path of the image to segment
    :param r: array with the different radius to use
    """
    images = {}

    for radius in r:
        segmented = study_image(img_path, radius, 2, make_5d=True, opt=True, plot_3d=False, save_stats=True)
        title = "r = " + str(radius)
        images[title] = segmented
    plot_dict(images)

    images = {}
    for radius in r:
        segmented = study_image(img_path, radius, 2, make_5d=False, opt=True, plot_3d=False, save_stats=True)
        title = "r = " + str(radius)
        images[title] = segmented
    plot_dict(images)


# debug_algorithm()
# compare_dimensions('../img/test_2.jpg', [10, 20, 30])

study_image('stop.jpg', 5, 4, make_5d=True, opt=True)
# study_image('../img/55075.jpg', 5, 4, make_5d=False, opt=True)

# study_image('../img/368078.jpg', 20, 4, make_5d=False, opt=True)
# study_image('../img/368078.jpg', 20, 4, make_5d=False, opt=False)
# study_image('../img/368078.jpg', 20, 4, make_5d=True, opt=True)
# study_image('../img/368078.jpg', 20, 4, make_5d=True, opt=False)

# study_image('../img/181091.jpg', 10, 4, make_5d=False, opt=True)
# study_image('../img/181091.jpg', 10, 2, make_5d=False, opt=True)
# study_image('../img/181091.jpg', 10, 1, make_5d=False, opt=True)