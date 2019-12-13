import cv2
import numpy as np

from advlane.filter import Filter
from advlane.lanes import Lanes
from advlane.warper import Warper


class Pipeline:

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.lanes = Lanes()
        self.filter = Filter()
        self.warper = None

    def add_info(self, image):
        left_curverad, right_curverad = self.lanes.measure_curvature()
        center = self.lanes.measure_center(image.shape)
        txt_l = 'left curve rad: ' + str(round(left_curverad)) + " m"
        txt_r = 'right curve rad: ' + str(round(right_curverad)) + " m"
        txt_m = 'off center: ' + str(round(center, 2)) + " m"
        image = cv2.putText(image, txt_l, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, txt_r, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.putText(image, txt_m, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def draw_data(self, image):
        add_lines = np.zeros_like(image)
        draw_x = np.polyval(self.lanes.last_left_fit, self.lanes.last_ploty)  # evaluate the polynomial
        draw_points_l = np.asarray([draw_x, self.lanes.last_ploty]).T.astype(np.int32)

        draw_x = np.polyval(self.lanes.last_right_fit, self.lanes.last_ploty)  # evaluate the polynomial
        draw_points_r = np.asarray([draw_x, self.lanes.last_ploty]).T.astype(np.int32)
        pts = np.concatenate((draw_points_r, np.flip(draw_points_l, 0)), axis=0)
        # cv2.polylines(add_lines, [pts], True, (0, 255, 0), 3)  # args: image, points, closed, color
        cv2.fillPoly(add_lines, [pts], (0, 255, 0))
        add_lines = self.warper.unwarp(add_lines)

        return add_lines

    def process_img(self, image_org):
        self.warper = Warper(image_org.shape)
        image = cv2.undistort(image_org, self.mtx, self.dist, None, self.mtx)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()
        filtered = self.filter.filter_lanes(image)
        warped = self.warper.warp(filtered)
        # plt.imshow(filtered, cmap='gray')
        # plt.show()
        # plt.imshow(warped, cmap='gray')
        # plt.show()

        if self.lanes.last_left_fit is None:
            out_img = self.lanes.fit_polynomial(warped)
            # plt.imshow(out_img, cmap='gray')
            # plt.show()
        else:
            self.lanes.search_around_poly(warped)

        add_lines = self.draw_data(image_org)
        # plt.imshow(cv2.cvtColor(add_lines, cv2.COLOR_BGR2RGB))
        # plt.show()
        result = cv2.addWeighted(image_org, 1, add_lines, 0.7, 0)

        result = self.add_info(result)
        return result
