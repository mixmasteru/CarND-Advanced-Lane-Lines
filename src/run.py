import os
import pickle
import cv2
import matplotlib.pyplot as plt
from advlane.pipeline import Pipeline

path = os.path.dirname(__file__)
pickle_file = os.path.join(path, 'assets/*.jpg')

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open(path + "/../assets/mtx_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

pipeline = Pipeline(mtx, dist)


fname = path + '/../assets/test_images/test5.jpg'
fname = path + '/../assets/test_images/straight_lines1.jpg'
image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

img = pipeline.process_img(image)
plt.imshow(img)
plt.show()