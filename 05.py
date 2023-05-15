import numpy as np
import cv2

img1 = cv2.imread("img_left.jpg")
img2 = cv2.imread("img_right.jpg")

detector = cv2.SIFT_create()
matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
k1, d1 = detector.detectAndCompute(img1, None)
k2, d2 = detector.detectAndCompute(img2, None)
match = cv2.BFMatcher()
matches = match.knnMatch(d2, d1, k = 2)

good = []
for m, n in matches:
    if m.distance < 0.8* n.distance:
        good.append(m)

MIN_MATCH_COUNT = 20
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([k2[m.queryIdx].pt for m in good])
    dst_pts = np.float32([k1[m.trainIdx].pt for m in good])
    H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
else:
    print('Not enought matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT))
    exit(1)     
    
img_warped = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0] + 500))
cv2.imwrite("img_warpped.jpg", img_warped)
img_stitched = img_warped.copy()
img_stitched[:img1.shape[0], :img1.shape[1]] = img1

def trim(frame):
    if np.sum(frame[0]) == 0:
        return trim(frame[1:])
    if np.sum(frame[-1]) == 0:
        return trim(frame[:-2])
    if np.sum(frame[:,0]) == 0:
        return trim(frame[:, 1:])
    if np.sum(frame[:,-1]) == 0:
        return trim(frame[:, :-2])
    return frame

img_stitched_trimmed = img_stitched
img_blur = cv2.blur(img_stitched_trimmed, (20, 20))
img_key = cv2.drawKeypoints(img2, k2, None)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor = None, flags=2)
img_match = cv2.drawMatches(img2, k2, img1, k1, good[:50], None, **draw_params)

cv2.imwrite("img_trimmed.jpg", img_stitched_trimmed)
cv2.imwrite("img_blur.jpg", img_blur)
cv2.imwrite("img_match.jpg", img_match)