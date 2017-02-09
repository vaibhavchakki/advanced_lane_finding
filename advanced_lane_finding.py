import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Camera import *
from Line import *


def draw(img):
   # Create an image to draw the lines on
   warp_zero = np.zeros_like(img).astype(np.uint8)
   color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

   # Recast the x and y points into usable format for cv2.fillPoly()
   ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
   pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
   pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
   pts = np.hstack((pts_left, pts_right))

   # Draw the lane onto the warped blank image
   cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

   # Warp the blank back to original image space using inverse perspective matrix (Minv)
   newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
   # Combine the result with the original image
   result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
   plt.imshow(result)

def main():
   camera = Camera()
   line = Line()
   camera.calibrate_camera(9, 6, "camera_cal/calibration*.jpg")
   original_image = mpimg.imread("test_images/straight_lines2.jpg")
   img = camera.apply_image_pipe(original_image)
   left_fit, right_fit = line.sliding_window_fit_plot(img)


if __name__ == "__main__":
   main()