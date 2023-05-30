import numpy as np


# NO OTHER IMPORTS ALLOWED


# Creates a greyscale histogram of the given image
def create_greyscale_histogram(img):
    """
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    """
    # img.ravel to convert the 2D image, which is a 2D matrix, into a 1D array, i.e. to smooth it out
    img_1d = img.ravel()

    # hist, _ indicates that the bin edges are discarded, important otherwise wrong shape of the histogram
    hist, _ = np.histogram(img_1d, bins=256, range=(0, 256))

    return hist


# Binarize an image with a given threshold
def binarize_threshold(img, t):
    """
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    """
    # If img > t True, then True is represented as 1 and False as 0, by multiplication by 255, is True
    # White and False Black
    img_bin = (img > t) * 255

    return img_bin


# Calculate p0 and p1
def p_helper(hist, theta: int):
    """
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    """
    # p0 as the sum of the histogram values from index 0 up to and including theta
    p0 = np.sum(hist[:theta + 1])

    # p1 as the sum of the histogram values from the index theta +1 to the end
    p1 = np.sum(hist[theta + 1:])

    return p0, p1


# Calculate mu0 and mu1
def mu_helper(hist, theta, p0, p1):
    """
    Compute mu0 and mu1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1

    mu Average of the elements of the histogram up to theta
    p Sum of elements in the subset
    + 10**-8 to avoid errors in zero division
    mu = sum of the element-wise multiplication in the range and the corresponding histogram values / sum of the
    elements in the subset
    """
    # Part of the histogram up to index theta + 1
    mu0 = np.sum(np.arange(theta + 1) * hist[:theta + 1]) / (p0 + 10 ** -8)

    # Ensure that there are enough elements in hist after theta + 1
    if theta + 1 < len(hist):
        # Part of the histogram to the end of the index theta + 1
        mu1 = np.sum(np.arange(theta + 1, len(hist)) * hist[theta + 1:]) / (p1 + 10 ** -8)
    else:
        mu1 = 0.0  # mu1 is zero if no elements in hist after theta + 1

    return mu0, mu1


# Calculate the threshold
def calculate_otsu_threshold(hist):
    """
    calculates theta according to otsus method
    :param hist: 1D array
    :return: threshold (int)
    """
    # Variables
    max_var = 0
    threshold = 0

    # Convert histogram to show probability distribution of pixels
    # --> sum(hist) = 1
    n = np.sum(hist)
    hist = hist / n

    # Go through all thetas
    # For each bin is assumed to be the threshold
    for theta in range(256):
        # Calculate p0 and o1
        p0, p1 = p_helper(hist, theta)

        # Calculate mu0 and mu1
        mu0, mu1 = mu_helper(hist, theta, p0, p1)

        # Calculate variance
        variance = p0 * p1 * (mu1 - mu0) ** 2

        # Update threshold, threshold = bin with the highest variance is the threshold of Otsu
        if variance > max_var:
            max_var = variance
            threshold = theta

    return threshold


# Apply otsu's method to an image
def otsu(img):
    """
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    """
    # Apply functions
    hist = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(hist)
    return binarize_threshold(img, threshold)
