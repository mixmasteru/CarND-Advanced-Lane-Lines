import cv2
import numpy as np


class Filter:

    def __init__(self):
        # Threshold x gradient
        self.grad_thresh_min = 20
        self.grad_thresh_max = 100
        # Threshold color channel
        self.s_thresh_min = 170
        self.s_thresh_max = 255
        # Kernel size
        self.ksize = 3
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([255, 25, 255])
        self.lower_yellow = np.array([15, 0, 70])
        self.upper_yellow = np.array([45, 255, 255])

    def abs_sobel_thresh(self, gray, orient='x'):
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= self.grad_thresh_min) & (scaled_sobel <= self.grad_thresh_max)] = 1

        return grad_binary

    def yellow_white(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        w_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        y_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        white_yellow_mask = cv2.bitwise_or(w_mask, y_mask)

        yellow_white = cv2.bitwise_and(image, image, mask=white_yellow_mask)
        # plt.imshow(cv2.cvtColor(yellow_white, cv2.COLOR_BGR2RGB))
        # plt.show()
        return white_yellow_mask

    def filter_lanes(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        white_yellow_mask = self.yellow_white(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        s_channel = hls[:, :, 2]

        sxbinary = self.abs_sobel_thresh(gray)

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh_min) & (s_channel <= self.s_thresh_max)] = 1

        # Combine the binary
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1) | (white_yellow_mask == 1)] = 1

        return combined_binary
