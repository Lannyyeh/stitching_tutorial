import cv2
import numpy as np


class Undistorter():
    def __init__(self):
        self.mtx = np.array([[256.37188889,   0,        324.39734975],
                [0, 244.65438081, 228.86609487],
                [0,        0,        1]])
        self.dist =  np.array([[[-0.025, -0.005, 0.001, 0, 0]], 
                               [[-0.025, -0.005, 0.001, 0, 0]],
                               [[-0.025, -0.005, 0.001, 0, 0]],
                               [[-0.015, -0.005, 0.001, 0.005, -0.001]]])

    def distort(self,frame_idx,img):
        dst = cv2.undistort(img, self.mtx, self.dist[frame_idx], None, self.mtx)
        return dst
