import os
import pickle

dirname = os.path.dirname(__file__)
pickle_file = os.path.join(dirname, 'assets/*.jpg')

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open(dirname + "/../assets/mtx_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
