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
    # img.ravel um das 2D Bild, was eine 2D Matrix ist in ein 1D Array umzuformen, also zu flatten
    img_1d = img.ravel()
    # hist, _ spezifiziert, dass die bin edges discarded werden, wichtig sonst falscher shape des Histogramms
    hist, _ = np.histogram(img_1d, bins=256, range=(0, 256))

    return hist


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}

    wenn img > t True, dann wird True als 1 repraesentiert und False als 0, durch Mulitplikation mit 255, ist True
    weiß und False Black
    '''
    img_bin = (img > t) * 255

    return img_bin


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''

    # p0 als Summe der Histogramm-Werte von Index 0 bis Theta inklusive
    p0 = np.sum(hist[:theta + 1])
    # p1 als Summe der Histogramm-Werte von Index Theta +1 bis zum Ende
    p1 = np.sum(hist[theta + 1:])

    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and mu1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1

    mu Mittelwert der Elemente im Histogramm bis zu theta
    p Summe der Elemente in der Teilmenge
    + 10**-8 um zero division error zu vermeiden
    mu = Summe der elementweisen Multiplikation im Bereich und den entsprechenden Histogrammwerten / Summe der Elemente
    in der Teilmenge
    '''

    # Teil des Histogramms bis zu Index theta + 1
    mu0 = np.sum(np.arange(theta + 1) * hist[:theta + 1]) / (p0 + 10**-8)

    # Sicherstellen, dass genuegend Elemente in hist nach theta + 1
    if theta + 1 < len(hist):
        # Teil des Histogramms bis zum Ende von Index theta + 1
        mu1 = np.sum(np.arange(theta + 1, len(hist)) * hist[theta + 1:]) / (p1 + 10**-8)
    else:
        mu1 = 0.0  # mu1 ist Null, wenn keine Elemente in hist nach theta + 1

    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method
    :param hist: 1D array
    :return: threshold (int)
    '''

    # Variablen
    max_var = 0
    threshold = 0
    # Histogramm umwandeln, damit es Wahrscheinlichkeitsverteilung der Pixel darstellt
    # --> sum(hist) = 1
    n = np.sum(hist)
    hist = hist / n

    # durch alle thetas iterieren
    # fuer jede bin davon ausgehen, dass sie der threshold ist
    for theta in range(256):
        # p0 und o1 berechnen
        p0, p1 = p_helper(hist, theta)
        # mu0 und mu1 berechnen
        mu0, mu1 = mu_helper(hist, theta, p0, p1)
        # Varianz berechnen
        variance = p0 * p1 * (mu1 - mu0) ** 2
        # Threshold updaten, threshold = bin mit hoechster Varianz ist Otsus threshold
        if variance > max_var:
            max_var = variance
            threshold = theta

    return threshold


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    hist = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(hist)
    return binarize_threshold(img, threshold)
