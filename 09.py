import numpy as np
import cv2 as cv

img_left = cv.imread('left.jpg')  #queryimage # left image
img_right = cv.imread('right.jpg') #trainimage # right image
img_left_epi = cv.imread("left_epi.jpg")
img_right_epi = cv.imread("right_epi.jpg")
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_left, None)
kp2, des2 = sift.detectAndCompute(img_right, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
imsize = (img_right.shape[1], img_right.shape[0])

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

_, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, imsize)

H1 = np.array(H1, dtype=np.float32)
H2 = np.array(H2, dtype=np.float32)

out_left = cv.warpPerspective(img_left_epi, H1, imsize)
out_right = cv.warpPerspective(img_right_epi, H2, imsize)

cv.imwrite("left_out.jpg", out_left)
cv.imwrite("right_out.jpg", out_right)


imgL = cv.imread('left_out.jpg', 0)
imgR = cv.imread('right_out.jpg', 0)
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

cv.imwrite("result.jpg", disparity)