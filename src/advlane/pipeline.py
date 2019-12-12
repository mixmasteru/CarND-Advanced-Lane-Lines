import cv2
import matplotlib.pyplot as plt
import numpy as np

from advlane.lanes import Lanes
from advlane.warper import Warper


class Pipeline:
    ksize = 3

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.lanes = Lanes()

    def abs_sobel_thresh(self, gray, orient='x', thresh=(0, 255)):
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    def mag_thresh(self, gray, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return mag_binary

    def dir_threshold(self, gray, thresh=(0, np.pi / 2)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)

        # Rescale to 8 bit
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return dir_binary

    def process_img(self, image_org):

        image = cv2.undistort(image_org, self.mtx, self.dist, None, self.mtx)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Apply each of the threshold functions
        gradx = self.abs_sobel_thresh(s_channel, orient='x', thresh=(20, 100))
        grady = self.abs_sobel_thresh(s_channel, orient='y', thresh=(20, 100))
        mag_binary = self.mag_thresh(s_channel, mag_thresh=(20, 100))
        dir_binary = self.dir_threshold(s_channel, thresh=(0.7, 1.3))

        out = np.zeros_like(dir_binary)
        out[((grady == 1) & (gradx == 1)) | ((mag_binary == 1) & (dir_binary == 0))] = 1

        points, warped = Warper.warp_img(out)
        plt.imshow(out, cmap='gray')
        plt.show()

        out_img, left_fit, right_fit = self.lanes.fit_polynomial(warped)
        plt.imshow(out_img, cmap='gray')
        plt.show()

        left_curverad, right_curverad = self.lanes.measure_curvature()
        print(left_curverad)
        print(right_curverad)
        left_fitx, right_fitx, ploty = self.lanes.search_around_poly(warped)
        left_curverad, right_curverad = self.lanes.measure_curvature()
        print(left_curverad)
        print(right_curverad)

        return image_org
