'''
Created on 22.10.2020
@author: Charly, Max
'''

import unittest
import cv2
import numpy as np
import os
from otsu import binarize_threshold, create_greyscale_histogram, otsu, calculate_otsu_threshold, p_helper, mu_helper

# 5% tolerance
RTOL = 0.05


class TestOtsu(unittest.TestCase):

    def setUp(self) -> None:
        assert os.path.exists('contrast.jpg')
        self.img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
        assert self.img.dtype == np.uint8

    def test_binarize_threshold(self):
        res = binarize_threshold(np.arange(100).reshape(10, 10), 99)
        res2 = binarize_threshold(np.arange(4).reshape(2, 2), 1)
        res3 = binarize_threshold(np.array([[1, 3, 7], [8, 8, 8], [1, 5, 6]]), 5)

        self.assertIsInstance(res, np.ndarray, msg='Use numpy arrays')
        self.assertTrue(np.all(res == 0), msg="smaller equal is black")
        self.assertTrue(np.all(res2 == np.array([[0, 0], [255, 255]])), msg="check your thresholding")
        self.assertTrue(np.all(res3 == np.array([[0, 0, 255], [255, 255, 255], [0, 0, 255]])),
                        msg="check your thresholding")

    def test_create_greyscale_histogram(self):
        res = create_greyscale_histogram(self.img)
        self.assertIsInstance(res, np.ndarray, msg="use numpy arrays")
        self.assertTrue(res.shape[0] == 256)
        self.assertTrue(np.all(res[:17] == 0))
        self.assertTrue(np.abs(res[160] - 8416) <= 2)

    def test_calculate_otsu_threshold(self):
        test_distribution = np.hstack((np.arange(5, 0, -1), np.zeros(246), np.arange(5)))
        test_distribution2 = np.hstack((np.zeros(123), np.arange(5, 0, -1), np.zeros(123), np.arange(5)))
        thres_1 = calculate_otsu_threshold(test_distribution)
        self.assertTrue(np.abs(thres_1 - 4) <= 5, msg='check your constraint for the new threshold')
        thres_2 = calculate_otsu_threshold(test_distribution2)
        self.assertTrue(np.abs(thres_2 - 127) <= 5)
        histogram = create_greyscale_histogram(self.img)
        self.assertTrue(np.abs(120 - calculate_otsu_threshold(histogram)) <= 5, msg='check your threshold')

    def test_p_helper(self):
        a1 = np.arange(2)
        p0_1, p1_1 = p_helper(a1 / np.sum(a1), 1)
        self.assertTrue(np.abs(1 - (p0_1 + p1_1)) < 0.2)
        self.assertNotIsInstance(p0_1, np.ndarray, msg='p0 and p1 are single values')
        self.assertEqual(p0_1, 1, msg='check the theta boundary for p0')
        self.assertEqual(p1_1, 0, msg='check the theta boundary for p1')
        a2 = np.arange(3)
        p0_2, p1_2 = p_helper(a2 / np.sum(a2), 1)
        self.assertTrue(np.abs(0.33 - p0_2) < 0.2, msg='check the theta boundary for p0')
        self.assertTrue(np.abs(0.66 - p1_2) < 0.2, msg='check the theta boundary for p1')
        a3 = np.arange(10)
        p0_3, p1_3 = p_helper(a3 / np.sum(a3), 3)
        self.assertTrue(np.abs(1 - (p0_3 + p1_3)) < 0.2)
        self.assertTrue(np.abs(0.13 - p0_3) < 0.1, msg='Check your boundary of theta for p0')
        self.assertTrue(np.abs(0.86 - p1_3) < 0.1, msg='Check your boundary of theta for p1')

    def test_mu_helper(self):
        a1 = np.arange(10)
        mu0_1, mu1_1 = mu_helper(a1, 4, 0.4, 0.6)
        self.assertNotIsInstance(mu0_1, np.ndarray, msg='mu is a single value variable')
        self.assertTrue(np.abs(mu0_1 - 75.0) <= 1, msg='check your mu0 boundary concerning theta')
        self.assertTrue(np.abs(mu1_1 - 425.0) <= 1, msg='check your mu1 range')
        self.assertFalse(np.abs(mu0_1 - 30.0) <= 3, msg='mu0 -> check your coefficient')
        self.assertFalse(np.abs(mu1_1 - 255.0) <= 3, msg='mu1 -> check your coefficient')
        self.assertFalse(np.abs(mu1_1 - 35.0) <= 3, msg='mu0 -> check which x you have to consider in the sum')
        self.assertFalse(np.abs(mu1_1 - 133.0) <= 3, msg='mu1 -> check which x you have to consider in the sum')
        self.assertFalse(np.abs(mu1_1 - 451.0) <= 3,
                         msg='mu1 -> check which x you have to consider in the sum -> sum borders')
        mu0_2, mu1_2 = mu_helper(np.arange(130), 129, 0.4, 0.6)
        self.assertTrue(np.abs(mu1_2) <= 1, msg='check your mu1')

    def test_otsu(self):
        bin_patch = otsu(self.img[:200, :400].copy())
        self.assertIsInstance(bin_patch, np.ndarray)
        self.assertTrue((bin_patch[90:110] == 255).all())
        self.assertEqual(bin_patch[50, 50], 0)

if __name__ == '__main__':
    unittest.main()
