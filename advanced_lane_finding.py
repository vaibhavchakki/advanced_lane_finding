import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Camera import *
from Line import *

def main():
   camera = Camera()
   line = Line()
   camera.calibrate_camera(9, 6, "camera_cal/calibration*.jpg")
   img = camera.apply_image_pipe("test_images/straight_lines2.jpg")
   line.sliding_window_fit_plot(img)


if __name__ == "__main__":
   main()