import glob
import os
import pickle
import matplotlib.pyplot as plt

import cv2
from moviepy.editor import VideoFileClip

from advlane.pipeline import Pipeline

path = os.path.dirname(__file__)
pickle_file = os.path.join(path, 'assets/*.jpg')

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
# If there is no pickle file, run cali first
dist_pickle = pickle.load(open(path + "/../assets/mtx_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

pipeline = Pipeline(mtx, dist)

test_img = os.path.join(path, '../assets/test_images/*.jpg')
images = glob.glob(test_img)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    print("processing " + fname)
    image = cv2.imread(fname)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    image = pipeline.process_img(image)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    cv2.imwrite(path + '/../output/'+os.path.basename(fname), image)

video = "project_video.mp4"
video_output = path + '/../output/' + video
clip1 = VideoFileClip(path + '/../assets/' + video).subclip(0, 10)
out_clip = clip1.fl_image(pipeline.process_img)  # NOTE: this function expects color images!!
out_clip.write_videofile(video_output, audio=False)
