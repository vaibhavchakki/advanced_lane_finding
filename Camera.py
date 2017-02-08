import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# NOTE:
# If we use cv2.imread to read the images we need to use BGR2GRAY conversion
# if we use mpimg.imread to read image, we need to used RGB2GRAY conversion


#
# Method to calibrate the camera. This method reads the calibration
# images and calibrate it
class Camera:
   #
   # Camera constructor
   def __init__(self):
      self._mtx = None
      self._dist = None
      self._num_x = None
      self._num_y = None

      #self._src = np.float32([ [527, 500],
      #                         [759, 500],
      #                         [260, 680],
      #                         [1044, 680] ])
      #self._dst = np.float32([ [260,  500],
      #                         [1044, 500],
      #                         [260,  680],
      #                         [1044, 680] ])

      self._src = np.float32([ [552, 480],
                               [735, 480],
                               [367, 612],
                               [938, 612] ])
      self._dst = np.float32([ [367, 480],
                               [938, 480],
                               [367, 612],
                               [938, 612] ])


      self._M   = cv2.getPerspectiveTransform(self._src, self._dst)
      self._M_i = cv2.getPerspectiveTransform(self._dst, self._src)

   def _readandConvertToGray(self, path):
      img = cv2.imread(path)
      return self._convertToGray(img)

   def _convertToGray(self, img):
      return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   #
   # Calibrate the camera
   def calibrate_camera(self, num_x, num_y, image_path):
      img = []
      images = glob.glob(image_path)

      obj_points = []
      image_points = []

      objp = np.zeros((num_x * num_y, 3), np.float32)
      objp[:,:2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

      for fname in images:
         gray = self._readandConvertToGray(fname)
         ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

         if ret == True:
            print("Found Corners for image %s" % fname)
            image_points.append(corners)
            obj_points.append(objp)
         else:
            pass
            #raise Exception("Unable to find corners for %s" % fname)

      assert (len(image_points) > 0)

      ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points,
                                                         (gray.shape[1], gray.shape[0]), None, None)
      self._mtx = mtx
      self._dist = dist

      return


   #
   # Un distort the image
   def undistort(self, img):
      return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)

   #
   # Apply Sobel on image
   def abs_sobel_thresh(self, img, orient='x', sobel_kernel = 3, thresh = (0, 255)):
      # 1) Convert to gray image
      gray = self._convertToGray(img)

      # 2) Take the derivative in x or y given orient = 'x' or 'y'
      sobel = None
      if orient == 'x':
         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
      else:
         sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

      # 3) Take the absolute value of the derivative or gradient
      sobel_abs = np.absolute(sobel)

      # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
      sobel_scaled = np.uint8(255. * sobel_abs / np.max(sobel_abs))

      # 5) Create a mask of 1's where the scaled gradient magnitude
      # is > thresh_min and < thresh_max
      binary_output = np.zeros_like(sobel_scaled)
      binary_output[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1

      return binary_output

   def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
      # 1) Convert to grayscale
      gray = self._convertToGray(img)

      # 2) Take the gradient in x and y separately
      sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
      sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

      # 3) Calculate the magnitude
      sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

      # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
      sobel_scaled = np.uint8(255. * sobel_mag / np.max(sobel_mag))

      # 5) Create a binary mask where mag thresholds are met
      binary_output = np.zeros_like(sobel_scaled)
      binary_output[(sobel_scaled >= mag_thresh[0]) & (sobel_scaled <= mag_thresh[1])] = 1

      # binary_output = np.copy(img) # Remove this line
      return binary_output

   def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
      # 1) Convert to grayscale
      gray = self._convertToGray(img)

      # 2) Take the gradient in x and y separately
      sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
      sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

      # 3) Take the absolute value of the x and y gradients
      abs_sobelx = np.absolute(sobel_x)
      abs_sobely = np.absolute(sobel_y)

      # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
      direction = np.arctan2(abs_sobely, abs_sobelx)

      # 5) Create a binary mask where direction thresholds are met
      binary_output = np.zeros_like(direction)
      binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

      return binary_output

   # Define a function that thresholds the S-channel of HLS
   # Use exclusive lower bound (>) and inclusive upper (<=)
   def hls_select(img, thresh=(0, 255)):
      # 1) Convert to HLS color space
      hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
      s_channel = hls_img[:, :, 2]
      binary_output = np.zeros_like(s_channel)
      binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

      return binary_output

   def warped_perspective(self, img):
      return cv2.warpPerspective(img, self._M, (img.shape[1], img.shape[0]))

   def apply_image_pipe(self, img_path, s_thresh=(185, 255), sx_thresh=(20, 100)):
      img = mpimg.imread(img_path)

      #
      # Undistort and apply Perspective transform
      undistort = self.undistort(img)
      warped_img = self.warped_perspective(undistort)

      # Convert to HSV color space and separate the V channel
      hsv = cv2.cvtColor(warped_img, cv2.COLOR_RGB2HLS).astype(np.float32)
      l_channel = hsv[:, :, 1]
      s_channel = hsv[:, :, 2]

      # Sobel x
      sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
      abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
      scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

      # Threshold x gradient
      sxbinary = np.zeros_like(scaled_sobel)
      sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

      # Threshold color channel
      s_binary = np.zeros_like(s_channel)
      s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

      # Stack each channel
      # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
      # be beneficial to replace this channel with something else.
      color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

      plt.imshow(color_binary)
      plt.show()
      return color_binary
