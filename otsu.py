import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    pass


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    pass


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    pass


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    pass


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1

    # TODO loop through all possible thetas

        # TODO compute p0 and p1 using the helper function

        # TODO compute mu and m1 using the helper function

        # TODO compute variance

        # TODO update the threshold
    pass


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    pass
