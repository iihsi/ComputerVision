import numpy as np
import cv2

img = cv2.imread("input.jpg")
height, width, _ = img.shape

pts1 = np.float32([[176,182], [750,534], [130,1240], [758,1062]])
pts2 = np.float32([[100,200], [850,200], [100,700], [850,700]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (1100, 870))

cv2.imwrite("output.jpg", dst)