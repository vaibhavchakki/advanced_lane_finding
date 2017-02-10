import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# NOTE:
# If we use cv2.imread to read the images we need to use BGR2GRAY conversion
# if we use mpimg.imread to read image, we need to used RGB2GRAY conversion

class Perspective:
    def __init__(self):
         self._src = np.float32([ [527, 500],
                                 [759, 500],
                                 [260, 680],
                                 [1044, 680] ])

         self._dst = np.float32([ [260,  500],
                                 [1044, 500],
                                 [260,  680],
                                 [1044, 680] ])

         #self._src = np.float32([[552, 480],
         #                        [735, 480],
         #                        [367, 612],
         #                        [938, 612]])
         #self._dst = np.float32([[367, 480],
         #                        [938, 480],
         #                        [367, 612],
         #                        [938, 612]])

         self._M = cv2.getPerspectiveTransform(self._src, self._dst)
         self._M_i = cv2.getPerspectiveTransform(self._dst, self._src)

    def warp(self, img):
        return cv2.warpPerspective(img, self._M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self._M_i, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


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
        objp[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

        for fname in images:
            gray = self._readandConvertToGray(fname)
            ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

            if ret == True:
                #print("Found Corners for image %s" % fname)
                image_points.append(corners)
                obj_points.append(objp)
            else:
                pass
                # raise Exception("Unable to find corners for %s" % fname)

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

    def binary_thershold(self, img, s_thresh=(150, 255), sx_thresh=(20, 60)):
        #
        # Undistort and apply Perspective transform
        undistort = self.undistort(img)
        warped_img = Perspective().warp(undistort)

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
        binary_output = np.zeros_like(sxbinary)
        binary_output[(s_binary == 1) | (sxbinary == 1)] = 1

        # plt.imshow(color_binary)
        # plt.show()
        return binary_output
