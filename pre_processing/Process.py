from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from stitching.feature_detector import FeatureDetector


def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    weir_imgs = ['./imgs/labc1_2.jpg']
    origin_img = cv.imread(weir_imgs[0])
    median_blur = cv.medianBlur(origin_img, 3)

    gray_img = cv.cvtColor(median_blur, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(
        gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    finder = FeatureDetector(detector='brisk')
    features = finder.detect_features(gray_img)
    keypoints_center_img = finder.draw_keypoints(gray_img, features)
    plot_image(keypoints_center_img, (15, 10))

    plot_image(origin_img)
    plot_image(median_blur)
    plot_image(thresh)
