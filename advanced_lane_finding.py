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

   ym_per_pix = 30 / 720  # meters per pixel in y dimension
   xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
   y_eval = np.max(ploty)

   #left_curverad, right_curverad = measure_curvature(result)
   #left_curverad, right_curverad = 0.0, 0.0
   # Fit new polynomials to x,y in world space
   #left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
   #right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
   # Calculate the new radii of curvature
   left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
       2 * left_fit[0])
   right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
       2 * right_fit[0])
   curvature = (left_curverad + right_curverad) / 2.

   left_dev = img.shape[0] * left_fit[0] ** 2 + img.shape[0] * left_fit[1] + left_fit[2]
   right_dev = img.shape[0] * right_fit[0] ** 2 + img.shape[0] * right_fit[1] + right_fit[2]
   deviation = ((left_dev + right_dev) / 2.) - (img.shape[1] / 2.)
   deviation = deviation * xm_per_pix

   cv2.putText(result, "Left Curve: {}, Right Curve: {}".format(left_curverad.round(2), right_curverad.round(2)),
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)

   cv2.putText(result, "Deviation: {}m".format(deviation.round(2)),
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=2)

   return result


def process_image(image, filename=None, save = 0):
   line = Line()
   undist = camera.undistort(image)

   if save == 1:
       cv2.imwrite("output_images/" + filename + "_undistort" + ".jpg", undist)

   binary_output = camera.binary_thershold(undist)

   if save == 1:
       cv2.imwrite("output_images/" + filename + "_binary" + ".jpg", binary_output * 200)

   warped_img = Perspective().warp(binary_output)

   if save == 1:
       cv2.imwrite("output_images/" + filename + "_warped" + ".jpg", warped_img * 200)

   left_fit, right_fit = line.sliding_window_fit_plot(warped_img)
   result =  draw(warped_img, undist, left_fit, right_fit)

   if save == 1:
       cv2.imwrite("output_images/" + filename + "_final" + ".jpg", result)

   return result

def process_video(video):
   in_video  = "{}.mp4".format(video)
   out_video = "{}_output.mp4".format(video)

   clip = VideoFileClip(in_video)
   output_clip = clip.fl_image(process_image)
   output_clip.write_videofile(out_video, audio=False)


if __name__ == "__main__":
   camera = Camera()
   camera.calibrate_camera(9, 6, "camera_cal/calibration*.jpg")
   #img = cv2.imread("test_images/test6.jpg")
   #output_img = process_image(img, filename="test6", save=1)
   process_video("project_video")