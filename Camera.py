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
         self._src = np.float32([ [580, 460],
                                 [700, 460],
                                 [260, 680],
                                 [1040, 680] ])

         self._dst = np.float32([ [260, 0],
                                 [1040, 0],
                                 [260, 680],
                                 [1040, 680] ])

         #self._src = np.float32( [ [270, 680],
         #                          [1050, 680],
         #                          [550, 480],
         #                          [740, 480]])
         #self._dst = np.float32( [[270, 680],
         #                          [1050, 680],
         #                         [270, 480],
         #                         [1050, 480]])


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

    def SobelOp(self, ch, orient='x', ksize=15, scaled = 1, absolute = 1):
        if orient == 'x':
            sobel = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize = ksize)
        else:
            sobel = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize = ksize)

        if absolute == 1:
            abs_sobel = np.absolute(sobel)
        else:
            return sobel

        if scaled == 1:
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        else:
            return abs_sobel

        return scaled_sobel

    def _color_mask(self, img, low = [0, 0, 0], high = [255, 255, 255]):
        mask = cv2.inRange(img, np.array(low), np.array(high))
        #binary = cv2.bitwise_and(img, img, mask = mask)
        binary = np.zeros_like(img[:, :, 0])
        binary[(mask != 0)] = 1
        return binary

    def yellow_mask(self, img):
        return self._color_mask(img, low=[0, 100, 100], high=[100, 255, 255])

    def white_mask(self, img):
        return self._color_mask(img, low=[0, 0, 210], high=[255, 60, 255])

    def dir_threshold(self, img, thresh=(0, np.pi/2), ksize=9):
        gray = self._convertToGray(img)

        sobel_x = self.SobelOp(gray, scaled=0, ksize=ksize)
        sobel_y = self.SobelOp(gray, orient='y', scaled=0, ksize=ksize)

        direction = np.arctan2(sobel_y, sobel_x)
        binary = np.zeros_like(direction)
        binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return binary

    def mag_threshold(self, img, thresh=(150, 255), ksize=9):
        gray = self._convertToGray(img)
        sobel_x = self.SobelOp(gray, scaled=0, absolute=0, ksize=ksize)
        sobel_y = self.SobelOp(gray, orient='y', scaled=0, absolute=0, ksize=ksize)
        sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        sobel_scaled = np.uint8(255. * sobel_mag / np.max(sobel_mag))

        binary = np.zeros_like(sobel_scaled)
        binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
        return binary

    #
    # Un distort the image
    def undistort(self, img):
        return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)


    def binary_thershold(self, img):
        #s_thresh = (150, 255)
        x_thresh = (60, 100)
        d_thresh = (0.7, 1.2)
        m_thresh = (50, 150)

        #s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #s_binary = np.zeros_like(s_channel)
        #s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        yellow_lane = self.yellow_mask(hsv)
        white_lane = self.white_mask(hsv)

        gray = self._convertToGray(img)
        grad_x = self.SobelOp(gray, ksize=15)
        grad_x_binary = np.zeros_like(gray)
        grad_x_binary[(grad_x >= x_thresh[0]) & (grad_x <= x_thresh[1])] = 1

        #dir_binary = self.dir_threshold(img, thresh=d_thresh, ksize=15)
        #mag_binary = self.mag_threshold(img, thresh=m_thresh, ksize=15)

        binary_output = np.zeros_like(gray)
        binary_output[(yellow_lane == 1) | (white_lane == 1) |
                      (grad_x_binary == 1) ] = 1
                      #((mag_binary == 1) & (dir_binary == 1))] = 1

        return binary_output