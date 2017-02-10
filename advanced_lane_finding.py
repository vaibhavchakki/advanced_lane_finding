import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from Camera import *
from Line import *


camera = None

def draw(img, undist, left_fit, right_fit):
   # Create an image to draw the lines on
   warp_zero = np.zeros_like(img).astype(np.uint8)
   color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

   # Recast the x and y points into usable format for cv2.fillPoly()
   ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
   left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
   right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]

   pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
   pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
   pts = np.hstack((pts_left, pts_right))

   # Draw the lane onto the warped blank image
   cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

   # Warp the blank back to original image space using inverse perspective matrix (Minv)
   newwarp = Perspective().unwarp(color_warp)
   # Combine the result with the original image
   result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
   #plt.imshow(result)
   #plt.show()

   return result


def process_image(image):
   line = Line()
   binary_output = camera.binary_thershold(image)
   undist = camera.undistort(image)
   left_fit, right_fit = line.sliding_window_fit_plot(binary_output)
   return draw(binary_output, undist, left_fit, right_fit)

def process_video(video):
   in_video  = "{}.mp4".format(video)
   out_video = "{}_output.mp4".format(video)

   clip = VideoFileClip(in_video)
   output_clip = clip.fl_image(process_image)
   output_clip.write_videofile(out_video, audio=False)


if __name__ == "__main__":
   camera = Camera()
   camera.calibrate_camera(9, 6, "camera_cal/calibration*.jpg")
   #img = mpimg.imread("test_images/test6.jpg")
   #img = process_image(img)
   process_video("project_video")