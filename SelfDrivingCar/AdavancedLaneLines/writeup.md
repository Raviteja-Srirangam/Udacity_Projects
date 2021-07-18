
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

M
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/ChessBoardDistortion.jpg "Undistorted"
[image2]: ./output_images/SampleRoad.jpg "Undistorted"
[image3]: ./output_images/GradientX.jpg "Gradient X"
[image4]: ./output_images/GradientY.jpg "Gradient Y"
[image5]: ./output_images/MagnitudeGrad.jpg "Gradient Mag"
[image6]: ./output_images/DirectionGrad.jpg "Gradient Dir"
[image7]: ./output_images/ColorGrad.jpg "Gradient Color"
[image8]: ./output_images/CombinedGrad.jpg "Gradient Combined"
[image9]: ./output_images/PerspectiveTransform.jpg "Perspective transform"
[image10]:./output_images/Histogram.jpg "Histogram"
[image11]:./output_images/SlidingWindows.jpg "Histogram"
[image12]:./output_images/PlottedLane.jpg "Lane Plot"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
  


### Step 1. Camera Calibration

I have defined a function to calibrate the camera to make the images free from distortion
    Function:calibrateCamera()
        input: a folder of images
        output: distortion matrix
How I did it?
    1)Generate ideal object points such as [0,0,0] ....[8,8,8]
    2)Get the images and read them using matplotlib.pyplot.imread
    3)Convert the image to gray scale by using cv2.cvtColor function
    4)Find the corners in each image using cv2.findChessboardCorners()
    5)Create image point using the found corners in the above step
    6)Find the distortion coefficient using cv2.calibrateCamera() using object points(Point1) and image        points(point 5)



### Step 2. Undistort the images:

I used the cv2.undistort function and the distortion matrix calculated from step 1 to undistort the image


The result can be seen below:
![alt text][image1] 
![alt text][image2]


### Step 3.Getting the Binary thresholded image
 1)Gradient in X and Y direction - absThresh()
     Used cv2.Sobel() functions to extract the gradients in x and y directions
 2)Gradient magnitude - magThresh()
    To extract the the pixels which are greater than the given threhold
 3)Gradient Direction - dirThresh()
    To extract the rate of change or slope of y with respect to x 
 4)Color threshold - sThresh()
     Used cv2.HLS() to convert the image to HLS scale and extracted the 'S' part greater than given threshold
 5)A final iterration where it combines all the the gradients in above steps - combinedThreshold()   
 
X direction:
![alt text][image3]
Y direction:
![alt text][image4]
Magnitude Gradient:
![alt text][image5]
Direction Gradient:
![alt text][image6]
Color Threshold:
![alt text][image7]
Combined Gradient
![alt text][image8]


### Step 4. Get the perspective transform of the image

I wrote the function Transform() to get the perspective transform of image
Explanation: 
    Inputs: img, source points(4) and destination points(4) 
    Outputs: transformed Image, Transform Matrix and inverse transform matrix
    Source points are chosen co-ordinates on the image and 
    For each source point we have a destination point to which it is shifted to.
    Source Point 1 -> Destination Point 1
    Source Point 2 -> Destination Point 2
    Source Point 3 -> Destination Point 3
    Source Point 4 -> Destination Point 4
    The points in between are shifted as per linear interpolation
    I did this using the function cv2.getTransform() ->returns Transform Matrix
    Using the transform matrix and cv2.warpPerspective(), we get the transformed image.
Perspective Transform:
![alt text][image9]

### Step 5. Getting the lane pixels and fitting them with a polynomial
This involves a series of steps which are explained below
1)Getting a histogram:
    Get the histogram of the lower half of the image.
    This is done via the function histogram(), it takes the perspcetive transformed image and 
    gives the histogram of the lower half.
Histogram:
![alt text][image10]

2)Detection of Lane pixels is done via function detectLines(). Input -> Image
  What is done?
      a)Get the histogram of the input image.
      b)Find the midpoint of it and also find the locations of maximum pixel value on either side of midpoint
      c)The index of maximum pixel on the left side of image -> Left reference point
        and for the right -> Right reference point
      d)We shall divide the entire left pixels and right pixels into a number of sliding windows.
      e)Margin for extension on each side of found pixels
      e)Height of each window -> Image height/no. of windows
      f)Iterate through each window:
          -Find 4 coordinates of each window ->lowY, highY, Left(Right)XMin, Left(Right)XMax
          -Get the lane pixels an left and right side for each window
          -If for any window, the number of pixels are greater than minpix(50), adapt the 
           Left and Right reference points
          -Concatenate all the left and right lane pixels found through all sliding windows
          -Fit them using np.polyfit() functions for both left and right side
 Sliding Windows:
 ![alt text][image11]
3)Detection of lanes in upcoming frames:
    What is done?
     a)We need nod go through the tedious process of sliding windows to find the lane pixels everytime.
       We can use the polynomials generated from previous images and use them to find new polynomials for the
       current image
     b)This is done in the function detectFurther(). It takes the img and polynomials from previous image
       If either left polynomial or the right is missed, it detects the lane pixels using sliding windows technique
       If not, it uses polynomials from previous frames and gets new polynomials for the current frame

       

### Step 6. Find the curvature and the postion of vehicle on the road:
I have written the function Curvature() -> 
 Input:Image and Lane polynomials
 Output:Left and Right Curvature, Position of vehicle on road
 What is done?
 a) Get the x and y co-ordinates of pixels on Left and Right side using Lane polynomials
 b)Convert the pixels to meters using Pixel to meter conversion on X and Y using 
   Xconv - 3.7/700(Lane width is 3.7 mts)
   Yconv - 30/720(The size of pixels in Y direction is 720 and the lane length is 30 mts)
 c)Get new x, y co-ordinates after pixel to meter conversion
 d)Curvature can be found using the formula:
   If polynomial is x = ay** 2 + by + c
   First derivative(j) - dx/dy =2Ax+B
   Second Derivative(k) - d2x/dy2 =2A
   ```python
   Curvature = (1+(2Ay+B)**2)**3/2/|2A|
   ```
 e)This is done for both left and right lines
 f)Postion of vehicle:
   Calculate the bottom position of both left and right lanes
   Get the center of image. 
   find the postion of vehicle using below code:
   ``` python

    left_lane_bottom = (leftfit[0]*y_eval)**2 + leftfit[0]*y_eval + leftfit[2]
    right_lane_bottom = (rightfit[0]*y_eval)**2 + rightfit[0]*y_eval + rightfit[2]
                       
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xConv #Convert to meters
    position = "left" if center < 0 else "right"
    center = "Vehicle is {:.2f}m {}".format(center, position)
   ```
### Step 7. Draw the Lane Corridor and display curvature and position of vehicle:

The function draw_on_image()
Inputs: Undistorted image(Step 2), perspective tranformed image(Step 4), left fit and right fit(Step 5), 
        transform matrix(step 4),Curvature of both sides(step 6) and Postion of vehicle(step 6)
Output: Iamge with Lane corridor and Curvature and Position fo vehicle

``` python
def draw_on_image(undist, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, center,show_values = False):
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    cv2.putText(result, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, '{}'.format(center), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
    return result
```
![alt text][image12]

### Step 8. Pipeline(Video)

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)
Here's a [link to my notebook](./Notebook.ipynb)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems which I faced during the project was to get the proper destination points to get the perspective transform of the image. Selecting the proper destination points would make the project detect the lanes even in curvy roads as in [hardchallenge](./test_videos/harder_challenge_video.mp4)
The pipeline most likely will fail if the curvature is too high and changing rapidly
The pipeline could be made robust 
    1)if detection of current lane pixels is checked for confidence
    2)if the lane pixels are averaged over the ones in previous frames.
