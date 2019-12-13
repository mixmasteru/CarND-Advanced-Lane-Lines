## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. 
* [x] Apply a distortion correction to raw images.
* [x] Use color transforms, gradients, etc., to create a thresholded binary image.
* [x] Apply a perspective transform to rectify binary image ("birds-eye view").
* [x] Detect lane pixels and fit to find the lane boundary.
* [x] Determine the curvature of the lane and vehicle position with respect to center.
* [x] Warp the detected lane boundaries back onto the original image.
* [x] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image00]: ./output/cali.png "cali"
[image0]: ./output/writeup1.png "input"
[image1]: ./output/writeup_undis.png "Undistorted"
[image2]: ./output/writeup_filtered.png "Filtered"
[image3]: ./output/writeup_warp.png "Warp"
[image4]: ./output/writeup_findlanes1.png "Find lanes window"
[image5]: ./output/writeup_findlanes2.png "Fit Visual"
[image6]: ./output/writeup_add.png "Add lane data"
[image7]: ./output/writeup_final.png "Final"
[video1]: ./output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points


---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the src/cali.py file 

+ apply findChessboardCorners to all calibration images
+ collect all object points and image points
+ run cv2.calibrateCamera with ith
+ store mtx & dist in pickle file

##### result
![alt text][image00]

### Pipeline (single images)

+ run.py calls the pipeline.process_img function
+ pipeline uses filter, lanes, warper classes which contain all logic
+ in run loading and saving of images and video is handeled

#### 1. Provide an example of a distortion-corrected image.

##### input
![alt text][image0]

##### undistorded
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

+ filtering is done in src/advlane/filter
+ generate a mask of white and yellow pixels in hsv space
+ apply an x sobel with an gradient threshold on a gray image
+ apply an threshold on the s channel of hls space
+ combine all of these to an binary

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp(image)`, in the warper class.
It takes the shape of the image and the image.  

```python
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
```

This resulted in the following image:

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane detection ins done in the lanes class.
The first image is does a sliding window approach with the help of histograms in fit_polynomial:

![alt text][image4]

The following images using the resulting lines and perform search_around_poly to find the lane pixels along these:  

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

* lanes.measure_curvature the calculation of the curvature is done on last_left_fit and last_right_fit
* lanes.measure_center also uses last_left_fit and last_right_fit to calculate 2 points of the lines and get the center of them, 
with the centre of the image shape this gives the offset

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

the pipeline.draw_data method takes the last_left_fit and last_right_fit to draw the lane via cv2.fillPoly

![alt text][image6]

it then unwarps the image with warper.unwarp and draws the textual data with pipeline.add_info 

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### Improvements
+ crop the part of the image with the engine hood
+ make the the whole pipeline save to changes in resolution of the image/video
+ the pipeline can't handle the challenge videos
+ I guess the filtering does not work well in these videos and also detects some pavement stuff als lanes 
