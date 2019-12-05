import cv2
import numpy as np


class Pipeline:
    ksize = 3

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

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

    def warp_img(self, image):
        # height, width, channels = img.shape
        vertices = np.array([[(200, image.shape[0]),
                              (int(image.shape[1] / 2) - 60, int(image.shape[0] / 2) + 100),
                              (int(image.shape[1] / 2) + 60, int(image.shape[0] / 2) + 100),
                              (image.shape[1] - 150, image.shape[0])]], dtype=np.int32)

        src = np.float32([[200, image.shape[0]],
                          [int(image.shape[1] / 2) - 60, int(image.shape[0] / 2) + 100],
                          [int(image.shape[1] / 2) + 60, int(image.shape[0] / 2) + 100],
                          [image.shape[1] - 150, image.shape[0]], ])
        xr = 300
        xl = 300
        dst = np.float32([[xr, image.shape[0]],
                          [xr, 0],
                          [image.shape[1] - xl, 0],
                          [image.shape[1] - xl, image.shape[0]]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        lines = cv2.polylines(image, vertices, True, (255, 120, 255), 3)
        return warped

    def process_img(self, image):
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
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
        out[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 0))] = 1

        out = self.warp_img(out)
        return out
