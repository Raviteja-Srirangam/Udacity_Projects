**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---


My pipeline consisted of 8 steps. 

 Step1:
     Get image from each frame in the video. Calculate the height and width of the image.

 Step 2:
     Convert the image to gray scale.
     This is done using the function grayscale(image). It takes the RGB image and returns the gray scale image.
     It returns img_gray
 
 Step 3:
     Smooth the image. This is done using gaussian_blur(img_gray, kernel_size)
     It returns img_blur
 
 Step 4:
     Use the smoothed image and pass it Canny detector function - canny(img_blur, low_threshold, high_threshold)
     This is used to extract the edges in the image.We can tune thresholds to improve edge detection
     It returns img_canny
 
 Step 5:
     This step is used to define the region of interest in the image to draw the lines.
     We have to define vertices based on the height and width of the image and extract the portion of image.
     This is done to concentrate only on Lane edges rather than all the edges in the image.
     This is accomplished by the function region_of_interest(img_canny,vertices)
     This returns roi_image. This shall be used further to determine lane boundaries.
 
 Step 6:
     In this step, we try to draw lane lines on the image by houghtransform.
     This is achieved by the function hough_lines(roiImage, rho, theta, threshold, min_line_len, max_line_gap)
     The parameters rho,theta, threshold, min_line_len and max_line_gap can be tuned to improve the detection.
     This returns lines. lines consists of all the line segments with specified parameters.
 
 Step 7:
     In this step, we shall get the slope of each line detected in the previous step.
     This is done using function getSlope(image,lines)
     If the slope is less than zero, it is left line. Otherwise, it is right line.
     We will get the avg slope of all the left and right lines.
     Using this we will find the coordinates to draw the lane lines.
     This is done using coordinates(image,avg) function. This returns array of coordinates for both left and right lanes n an     image.
     The getSlope() then returns both left and right lines as an array->final_lines.
  
 Step 8:
      We shall use drawLines(image, final_lines)
          image -> Colour image extracted from each frame of video
          final_lines-> the return of getSlope()
      drawLines() shall draw final_lines on image.


This shall be called for every frame in the input video.
      
![alt text][]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lanes are curvy. The detetion shall not be 100% accurate.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to make this accurate for lane detection in curvy roads by extraction the RegionofInterest properly.
