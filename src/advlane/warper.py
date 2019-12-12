import cv2
import numpy as np
import matplotlib.pyplot as plt


class Warper:

    @staticmethod
    def warp_img(image):
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

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        dst = np.around(dst).astype(int)
        return [dst], warped

    def unwarp_img(image):
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

        M = cv2.getPerspectiveTransform(dst, src)
        unwarped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        dst = np.around(dst).astype(int)
        return unwarped
