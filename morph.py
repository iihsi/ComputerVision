import cv2
import numpy as np
import sys
import dlib
from imutils import face_utils


def Face_landmarks(img_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    img = cv2.imread(img_path)
    size = img.shape
    rects = detector(img, 0)
    if len(rects) > 2 or len(rects) < 1:
        sys.exit(1)
    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        points = shape.tolist()
        points.append([0, 0])
        points.append([int(size[1]-1), 0])
        points.append([0, int(size[0]-1)])
        points.append([int(size[1]-1), int(size[0]-1)])
        points.append([int(size[1]/2), 0])
        points.append([0, int(size[0]/2)])
        points.append([int(size[1]/2), int(size[0]-1)])
        points.append([int(size[1]-1), int(size[0]/2)])
        cv2.destroyAllWindows()
        return points


def Face_delaunay(rect, points1, points2, alpha):
    points = []
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))
    triangles, delaunay = calculateDelaunayTriangles(rect, points)
    cv2.destroyAllWindows()
    return triangles, delaunay


def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p) 
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    pt = []
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        if (rectContains(rect, pt1) and rectContains(rect, pt2) and 
            rectContains(rect, pt3)):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if ((abs(pt[j][0] - points[k][0]) < 1.0 and
                         abs(pt[j][1] - points[k][1]) < 1.0)):
                        ind.append(k)    
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        pt = []
    return triangleList,delaunayTri


def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


def Face_morph(img1, img2, img, tri1, tri2, tri, alpha) :
    r1 = cv2.boundingRect(np.float32([tri1]))
    r2 = cv2.boundingRect(np.float32([tri2]))
    r = cv2.boundingRect(np.float32([tri]))
    t1Rect = []
    t2Rect = []
    tRect = []
    for i in range(0, 3):
        tRect.append(((tri[i][0] - r[0]),(tri[i][1] - r[1])))
        t1Rect.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
        t2Rect.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    warpMat1 = cv2.getAffineTransform(np.float32(t1Rect), np.float32(tRect))
    warpMat2 = cv2.getAffineTransform(np.float32(t2Rect), np.float32(tRect))
    size = (r[2], r[3])
    warpImage1 = cv2.warpAffine(img1Rect, warpMat1, (size[0], size[1]), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT_101)
    warpImage2 = cv2.warpAffine(img2Rect, warpMat2, (size[0], size[1]), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT_101)
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], 
    r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask


def betweenPoints(point1, point2, alpha) :
    points = []
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))
    return points


if __name__ == '__main__' :
    file1 = 'kaz.jpg'
    file2 = 'kar.jpg'
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    size = img1.shape
    rect = (0, 0, size[1], size[0])
    points1 = Face_landmarks(file1)
    points2 = Face_landmarks(file2)
    for cnt in range(1, 100):
        alpha = cnt * 0.01
        points = betweenPoints(points1, points2, alpha)
        triangles, delaunay = Face_delaunay(rect, points1, points2, alpha)
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
        for (i, (x, y, z)) in enumerate(delaunay):
            tri1 = [points1[x], points1[y], points1[z]]
            tri2 = [points2[x], points2[y], points2[z]]
            tri = [points[x], points[y], points[z]]
            Face_morph(img1, img2, imgMorph, tri1, tri2, tri, alpha)
        imgMorph = np.uint8(imgMorph)
        print(cnt)
        cv2.imwrite('cou-%s.jpg' % str(cnt).zfill(3), imgMorph)