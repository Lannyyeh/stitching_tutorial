import cv2

# 打开视频文件
video = cv2.VideoCapture('./testolabc4.avi')

i = 1
while True:
    # 读取每帧图片
    _, im = video.read()
    if im is None:
        print('vedio playing ended')
        break
    i = i+1
    filename = './lab4_frames/labc4_'+str(i)+'.jpg'
    cv2.imwrite(filename, im)

    cv2.imshow('vedio playing', im)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
