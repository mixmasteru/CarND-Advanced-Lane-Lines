import glob
import os
import pickle

import cv2
import numpy as np

# grid counts
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, '../camera_cal/*.jpg')
images = glob.glob(path)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    print(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print("corners_found")
        # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

path = os.path.join(dirname, '../camera_cal/calibration1.jpg')
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# save calibration
dist_pickle = {"mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open(dirname + "/../assets/mtx_dist_pickle.p", "wb"))

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imshow('dst', dst)
cv2.waitKey(5000)
