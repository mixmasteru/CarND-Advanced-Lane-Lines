import cv2
import numpy as np


class Warper:

    def __init__(self, shape):
        # height, width, channels = shape
        xr = 300
        xl = 300
        self.src = np.float32([[200, shape[0]],
                               [int(shape[1] / 2) - 60, int(shape[0] / 2) + 100],
                               [int(shape[1] / 2) + 60, int(shape[0] / 2) + 100],
                               [shape[1] - 150, shape[0]], ])

        self.dst = np.float32([[xr, shape[0]],
                               [xr, 0],
                               [shape[1] - xl, 0],
                               [shape[1] - xl, shape[0]]])

    def warp(self, image):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        # dst = np.around(self.dst).astype(int)
        return warped

    def unwarp(self, image):
        M = cv2.getPerspectiveTransform(self.dst, self.src)
        unwarped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        # dst = np.around(self.dst).astype(int)
        return unwarped

    def draw_regions(self, image):
        vertices = np.array([[(200, image.shape[0]),
                              (int(image.shape[1] / 2) - 60, int(image.shape[0] / 2) + 100),
                              (int(image.shape[1] / 2) + 60, int(image.shape[0] / 2) + 100),
                              (image.shape[1] - 150, image.shape[0])]], dtype=np.int32)
