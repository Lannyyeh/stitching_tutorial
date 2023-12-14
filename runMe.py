from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import time
from pathlib import Path
from pre_processing.Undistort import Undistorter
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.subsetter import Subsetter
from stitching.warper import Warper
from stitching.timelapser import Timelapser
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender


def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]


if __name__ == "__main__":

    # start = time.time()
    img1=cv.imread('./imgs/labc1_2.jpg')
    img2=cv.imread('./imgs/labc2_2.jpg')
    img3=cv.imread('./imgs/labc3_2.jpg')
    img4=cv.imread('./imgs/labc4_2.jpg')

    m_distorter=Undistorter()
    img1=m_distorter.distort(0,img1)
    img2=m_distorter.distort(1,img2)
    img3=m_distorter.distort(2,img3)
    img4=m_distorter.distort(3,img4)
    
    weir_imgs = list([img1,img2,img3,img4])
    images = Images.of(weir_imgs)

    medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
    low_imgs = list(images.resize(Images.Resolution.LOW))
    final_imgs = list(images.resize(Images.Resolution.FINAL))

    finder = FeatureDetector(detector='brisk')
    features = [finder.detect_features(img) for img in medium_imgs]

    matcher = FeatureMatcher()
    matches = matcher.match_features(features)

    matcher.get_confidence_matrix(matches)
    subsetter = Subsetter(confidence_threshold=0.75,
                          matches_graph_dot_file=None)
    dot_notation = subsetter.get_matches_graph(images.names, matches)
    print(dot_notation)

    indices = subsetter.get_indices_to_keep(features, matches)

    medium_imgs = subsetter.subset_list(medium_imgs, indices)
    low_imgs = subsetter.subset_list(low_imgs, indices)
    final_imgs = subsetter.subset_list(final_imgs, indices)
    features = subsetter.subset_list(features, indices)
    matches = subsetter.subset_matches(matches, indices)

    images.subset(indices)

    print(images.names)
    print(matcher.get_confidence_matrix(matches))
    print()

    camera_estimator = CameraEstimator(estimator='homography')
    camera_adjuster = CameraAdjuster()
    wave_corrector = WaveCorrector()

    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    cameras = wave_corrector.correct(cameras)

    warper = Warper()
    warper.set_scale(cameras)

    low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
    # since cameras were obtained on medium imgs
    camera_aspect = images.get_ratio(
        Images.Resolution.MEDIUM, Images.Resolution.LOW)

    warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
    warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
    low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

    final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

    warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
    warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

    # timelapser = Timelapser('as_is')
    # timelapser.initialize(final_corners, final_sizes)

    cropper = Cropper()

    mask = cropper.estimate_panorama_mask(
        warped_low_imgs, warped_low_masks, low_corners, low_sizes)

    lir = cropper.estimate_largest_interior_rectangle(mask)
    # lir = cropper.estimate_largest_interior_rectangle(mask)

    low_corners = cropper.get_zero_center_corners(low_corners)
    rectangles = cropper.get_rectangles(low_corners, low_sizes)
    overlap = cropper.get_overlap(rectangles[1], lir)
    intersection = cropper.get_intersection(rectangles[1], overlap)

    cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

    cropped_low_masks = list(cropper.crop_images(warped_low_masks))
    cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
    low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

    # since lir was obtained on low imgs
    lir_aspect = images.get_ratio(
        Images.Resolution.LOW, Images.Resolution.FINAL)
    cropped_final_masks = list(
        cropper.crop_images(warped_final_masks, lir_aspect))
    cropped_final_imgs = list(
        cropper.crop_images(warped_final_imgs, lir_aspect))
    final_corners, final_sizes = cropper.crop_rois(
        final_corners, final_sizes, lir_aspect)

    # timelapser = Timelapser('as_is')
    # timelapser.initialize(final_corners, final_sizes)

    seam_finder = SeamFinder()

    seam_masks = seam_finder.find(
        cropped_low_imgs, low_corners, cropped_low_masks)
    seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(
        seam_masks, cropped_final_masks)]

    # start_blend=time.time()
    compensator = ExposureErrorCompensator()

    compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

    compensated_imgs = [compensator.apply(idx, corner, img, mask)
                        for idx, (img, mask, corner)
                        in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]

    
    
    blender = Blender()
    blender.prepare(final_corners, final_sizes)
    for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
        blender.feed(img, mask, corner)
    panorama, _ = blender.blend()

    # end_blend = time.time()
    # print('exec time: %s seconds' % (end_blend-start_blend))
    
    # plot_image(panorama, (20, 20))
    print("First test has been down")

    video_1 = cv.VideoCapture('./testolabc1.avi')
    video_2 = cv.VideoCapture('./testolabc2.avi')
    video_3 = cv.VideoCapture('./testolabc3.avi')
    video_4 = cv.VideoCapture('./testolabc4.avi')

    frame_count=0
    while 1:
        _, frame_1 = video_1.read()
        _, frame_2 = video_2.read()
        _, frame_3 = video_3.read()
        _, frame_4 = video_4.read() 
        
        frame_1=m_distorter.distort(0,frame_1)
        frame_2=m_distorter.distort(1,frame_2)
        frame_3=m_distorter.distort(2,frame_3)
        frame_4=m_distorter.distort(3,frame_4)

        weir_imgs=list([frame_1,frame_2,frame_3,frame_4])

        images = Images.of(weir_imgs)    
       
        final_imgs = list(images.resize(Images.Resolution.FINAL))  
        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
                
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(cropped_final_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        cv.imshow("test",panorama)
        cv.waitKey(1)
        frame_count=frame_count+1
        print("Now playing at : ",frame_count)

    cv.destroyAllWindows()
