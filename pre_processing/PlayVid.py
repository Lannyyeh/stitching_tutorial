import cv2 as cv

# 打开视频文件
video_1 = cv.VideoCapture('./../testolabc1.avi')
video_2 = cv.VideoCapture('./../testolabc2.avi')
video_3 = cv.VideoCapture('./../testolabc3.avi')
video_4 = cv.VideoCapture('./../testolabc4.avi')

i = 1
while True:
    # 读取每帧图片
    _, frame_1 = video_1.read()
    _, frame_2 = video_2.read()
    _, frame_3 = video_3.read()
    _, frame_4 = video_4.read()
    if frame_1 is None:
        print('vedio playing ended')
        break
    i = i+1

    cv.imshow('vedio playing 1', frame_1)
    cv.imshow('vedio playing 2', frame_2)
    cv.imshow('vedio playing 3', frame_3)
    cv.imshow('vedio playing 4', frame_4)
    cv.waitKey(1)

video_1.release()
video_2.release()
video_3.release()
video_4.release()
cv.destroyAllWindows()
