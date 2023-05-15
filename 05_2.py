import cv2

img = cv2.imread("img_trimmed.jpg")
#img_blur = cv2.bilateralFilter(img, 9, 100, 100)
blur = cv2.blur(img,(20,20))
cv2.imwrite("img_blur2.jpg", blur)