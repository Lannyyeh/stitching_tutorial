import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images=glob.glob("D://Files//3-2//009_3F//Code//images/*.jpg")
i=0

for image in images:
    img= cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    #print(corners)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  
        
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (7, 7), corners, ret)  
        i+=1;
        cv2.imwrite('conimg'+str(i)+'.jpg', img)
        cv2.waitKey(1500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

f = open('param.txt','w')
print("mtx= "+str(mtx),file=f)
print("dist= "+str(dist),file=f)
f.close()

print(type(mtx))
print(type(dist))





