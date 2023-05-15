import numpy as np
import cv2 as cv
from PIL import Image
import open3d as o3d


def Homography(pre_ps, tfd_ps):
    assert (pre_ps.shape[0] == tfd_ps.shape[0]) & (tfd_ps.shape[0] == 4)
    hmatrix = np.ones((9, 1))
    A = np.zeros((8, 8))
    for i in range(4):
        a_ind = i * 2
        b_ind = i * 2 + 1
        A[a_ind, 0] = pre_ps[i, 0]
        A[a_ind, 1] = pre_ps[i, 1]
        A[a_ind, 2] = 1.
        A[a_ind, 6] = -(pre_ps[i, 0] * tfd_ps[i, 0])
        A[a_ind, 7] = -(pre_ps[i, 1] * tfd_ps[i, 0])
        A[b_ind, 3] = pre_ps[i, 0]
        A[b_ind, 4] = pre_ps[i, 1]
        A[b_ind, 5] = 1.
        A[b_ind, 6] = -(pre_ps[i, 0] * tfd_ps[i, 1])
        A[b_ind, 7] = -(pre_ps[i, 1] * tfd_ps[i, 1])
    tfd_ps_flatten = tfd_ps.reshape(8, 1)
    A_inv = np.linalg.inv(A)
    hmatrix[:8] = np.dot(A_inv, tfd_ps_flatten)
    hmatrix = hmatrix.reshape((3, 3))
    return hmatrix


def CalcLineSegmentPoints(img, horizontal_id):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 200)
    cv.imwrite('edges_ex01.jpg', edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 110)
    lines_seg = np.zeros((lines.shape[0], 5))
    for i, line in enumerate(lines):
        rho, theta = line[0]
        alpha = np.pi / 18
        beta = np.pi / 18
        a = np.cos(theta)
        b = np.sin(theta)

        if (np.abs(np.cos(theta)) < np.cos(np.pi/2 - beta)):
            line_id = horizontal_id  # horizontal_id is decided by images
        elif (np.abs(np.cos(theta)) <= np.abs(np.cos(alpha))) and (np.cos(theta) >= np.cos(np.pi/2 - beta)):
            line_id = 1  # y lines
        elif np.abs(np.cos(theta)) > np.abs(np.cos(alpha)):
            line_id = 0  # z lines
        else:
            line_id = 2  # x lines
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines_seg[i, :] = np.array([x1, y1, x2, y2, line_id])
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv.imwrite('linesDetected_ex01.jpg', img)
    print("lines_seg")
    print(lines_seg.shape)
    return lines_seg


def DivideLines(lines, img):
    lines = lines.tolist()
    x_list = []
    y_list = []
    z_list = []
    for line in lines:
        if line[4] == 1:
            x_list.append(line[0:4])
        elif line[4] == 2:
            y_list.append(line[0:4])
        elif line[4] == 0:
            z_list.append(line[0:4])
    x_n = len(x_list)
    y_n = len(y_list)
    z_n = len(z_list)
    if x_n != 0:
        for i in range(x_n):
            cv.line(img, (int(x_list[i][0]), int(x_list[i][1])),
                    (int(x_list[i][2]), int(x_list[i][3])), (255, 255, 0), 1)
        x_list = np.array(x_list).reshape(x_n, 4)
    if y_n != 0:
        for i in range(y_n):
            cv.line(img, (int(y_list[i][0]), int(y_list[i][1])),
                    (int(y_list[i][2]), int(y_list[i][3])), (255, 0, 0), 1)
        y_list = np.array(y_list).reshape(y_n, 4)
    if z_n != 0:
        for i in range(z_n):
            cv.line(img, (int(z_list[i][0]), int(z_list[i][1])),
                    (int(z_list[i][2]), int(z_list[i][3])), (0, 255, 255), 1)
        z_list = np.array(z_list).reshape(z_n, 4)
    return x_list, y_list, z_list


def CalcLinesEquation(lines, img):
    h, w, c = img.shape
    weight = ((h + w) / 2) / 2
    l_num = lines.shape[0]
    lines = lines.reshape(l_num, 1, 2, 2)
    e_1 = lines[:, 0, 0].astype(np.float32)
    e_2 = lines[:, 0, 1].astype(np.float32)
    # coordinate transformation
    e_1[:, 0] -= w / 2.
    e_2[:, 0] -= w / 2.
    e_1[:, 1] -= h / 2.
    e_2[:, 1] -= h / 2.
    # Homogeneous coordinate transformation
    e_1 *= weight
    e_2 *= weight
    W = np.ones((l_num, 1)) * weight
    e_1 = np.append(e_1, W, axis=1)
    e_2 = np.append(e_2, W, axis=1)
    # Line equation
    lines_q = np.zeros((l_num, 3))
    for i in range(l_num):
        lines_q[i, :] = np.cross(e_1[i, :].reshape(1, 3), e_2[i, :].reshape(1, 3))
    return lines_q


def CalcVanishingPoints(lines):
    v_ps = []
    v_dis = []
    l_num = lines.shape[0]
    print("l_num")
    print(l_num)
    for i in range(l_num - 1):
        for j in range(i + 1, l_num):
            c = lines[i, 0] * lines[j, 1] - lines[i, 1] * lines[j, 0]            
            if abs(c) <= 0.01:
                v_ps.append((np.cross(lines[i, :], lines[j, :])).tolist())
    # From homogeneous coordinate to image coordinate
    v_ps = np.array(v_ps)
    v_num = len(v_ps)
    for i in range(v_num):
        v_ps[i, :] /= v_ps[i, 2]
    # Sum of distance from points to lines
    for point in v_ps:
        d_sum = 0
        for line in lines:
            d_sum += CalcDistance(line, point)
        v_dis.append(d_sum)
    v_dis = np.array(v_dis)
    index = v_dis.argmin()
    v = v_ps[index, :]
    return v


def CalcDistance(line, point):  # line ax+by+c=0 point(x0,y0)
    numer = abs(line[0] * point[0] + line[1] * point[1] + line[2])
    denom = np.sqrt(pow(line[0], 2) + pow(line[1], 2))
    return numer / denom


def VanishingPoint(lines, img):
    h, w, c = img.shape
    lines_q = CalcLinesEquation(lines, img)
    v = CalcVanishingPoints(lines_q)
    v[0] += w / 2.0
    v[1] += h / 2.0
    return v.astype(np.int32)


# show vanishing point
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def pil2cv(image):
    # PIL to OpenCV
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # monochrome
        pass
    elif new_image.shape[2] == 3:  # color
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # transparent
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def cv2pil(image):
    # OpenCV to PIL
    new_image = image.copy()
    if new_image.ndim == 2:  # monochrome
        pass
    elif new_image.shape[2] == 3:  # color
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # transparent
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def CalcZ_alpha(v_line, Zr, V):    
    src_pts = [[268, 644], [264, 164]]
    Br = np.array([src_pts[0][0], src_pts[0][1], 1]).reshape(1, 3)
    Tr = np.array([src_pts[1][0], src_pts[1][1], 1]).reshape(1, 3)
    num = -np.linalg.norm(np.cross(Br, Tr))
    den = Zr * (np.dot(v_line, Br.T)[0, 0]) * np.linalg.norm(np.cross(V, Tr))
    alpha = num / den
    return alpha


def Make2dGrid(w_max, h_max, z_max, non_plane_num, scale, P):
    w = int(w_max / scale)
    h = int(h_max / scale)
    s = h * w
    axis1 = np.arange(w)
    axis1 = np.tile(axis1, h).reshape(1, s)
    axis2 = np.arange(h)
    axis2 = (np.tile(axis2, w).reshape(w, h)).T
    axis2 = axis2.reshape(1, s)
    plane = np.stack([axis1[0, :], axis2[0, :]], 1)
    t_plane = np.stack([axis1[0, :], axis2[0, :]], 1) * scale
    zeros = np.zeros((1, 1))
    ones = np.ones((s, 1))
    plane = np.insert(plane, non_plane_num, zeros/scale, axis=1)
    t_plane = np.insert(t_plane, non_plane_num, zeros, axis=1)
    t_plane = np.append(t_plane, ones, axis=1)
    pts_2d = np.dot(P, t_plane.T)
    w = (pts_2d[2, :].T).reshape(s, 1)
    pts_2d = pts_2d.T / w
    pts_2d = pts_2d[:, :2]
    return pts_2d, plane


def Make3dGrid(w_max, h_max, z_max, non_plane_num, scale, P):
    w = int(w_max / scale)
    h = int(h_max / scale)
    d = int(z_max / scale)
    s = h * w
    all_pts_2d = np.array([])
    all_plane = np.array([])
    axis1 = np.arange(w)
    axis1 = np.tile(axis1, h).reshape(1, s)
    axis2 = np.arange(h)
    axis2 = (np.tile(axis2, w).reshape(w, h)).T
    axis2 = axis2.reshape(1, s)
    for i in range(d):
        plane = np.stack([axis1[0, :], axis2[0, :]], 1)
        t_plane = np.stack([axis1[0, :], axis2[0, :]], 1) * scale
        zeros = np.ones((1, 1)) * i
        ones = np.ones((s, 1))
        plane = np.insert(plane, non_plane_num, zeros, axis=1)
        t_plane = np.insert(t_plane, non_plane_num, zeros*scale, axis=1)
        t_plane = np.append(t_plane, ones, axis=1)
        pts_2d = np.dot(P, t_plane.T)
        w = (pts_2d[2, :].T).reshape(s, 1)
        pts_2d = pts_2d.T / w
        pts_2d = pts_2d[:, :2]
        if i == 0:
            all_pts_2d = pts_2d
            all_plane = plane
        else:
            all_pts_2d = np.vstack([all_pts_2d, pts_2d])
            all_plane = np.vstack([all_plane, plane])
    return all_pts_2d, all_plane


def CalcNearPoints(pt):
    # retrun 4 points for bilinear interpolation
    a = int(pt[0] - 0.5)
    b = int(pt[0] + 0.5)
    c = int(pt[1] - 0.5)
    d = int(pt[1] + 0.5)
    x1, x4 = a, a
    x2, x3 = b, b
    y1, y2 = d, d
    y3, y4 = c, c
    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def BilinearInterpolation(pts, img):
    values = []
    for pt in pts:
        nr_pts = CalcNearPoints(pt)
        a, b = img[nr_pts[0][1], nr_pts[0][0], :], img[nr_pts[1][1], nr_pts[1][0], :]
        c, d = img[nr_pts[2][1], nr_pts[2][0], :], img[nr_pts[3][1], nr_pts[3][0], :]
        d1, d2 = abs(pt[0] - nr_pts[0][0]), abs(pt[0] - nr_pts[1][0])
        d3, d4 = abs(pt[1] - nr_pts[0][1]), abs(pt[0] - nr_pts[3][1])
        p1_2 = a*(d1/(d1+d2)) + b*(d2/(d1+d2))
        p3_4 = d*(d1/(d1+d2)) + c*(d2/(d1+d2))
        value = p1_2*(d3/(d3+d4)) + p3_4*(d4/(d3+d4))
        values.append(value)
    return values


def Convert3dTo2d(x_max, y_max, z_max, scale, P, img):
    xz_2d, xz_3dpts = Make2dGrid(x_max, z_max, y_max, 1, scale, P)
    yz_2d, yz_3dpts = Make2dGrid(y_max, z_max, x_max, 0, scale, P)
    xy_2d, xy_3dpts = Make2dGrid(x_max, y_max, z_max, 2, scale, P)
    # bilinear interpolation
    xz_values = np.array(BilinearInterpolation(xz_2d, img))
    yz_values = np.array(BilinearInterpolation(yz_2d, img))
    xy_values = np.array(BilinearInterpolation(xy_2d, img))
    _3dpts = np.vstack([xz_3dpts, yz_3dpts])
    _3dpts = np.vstack([_3dpts, xy_3dpts])
    values = np.vstack([xz_values, yz_values])
    values = np.vstack([values, xy_values])
    print("xy_3dpts")
    print(xy_3dpts)
    # keep as 3D data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_3dpts)
    pcd.colors = o3d.utility.Vector3dVector(values/255.)
    o3d.estimate_normals(pcd, search_param=o3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd], "points", 1280, 960)
    o3d.io.write_point_cloud("sync_02.ply", pcd)


if __name__ == '__main__':
    # Input an image
    img = cv.imread('svm4.png')
    img2 = img.copy()
    h, w, c = img.shape
    
    # Edge and line detection
    lines = CalcLineSegmentPoints(img, 2)
    x_lines, y_lines, z_lines = DivideLines(lines, img)
    v_x, v_y, v_z = [], [], []
    top, right, bottom, left = 10, 10, 10, 10
    color = (255, 255, 255)
    if not (type(x_lines) is list):
        v_x = VanishingPoint(x_lines, img)
        left = abs(v_x[0]) + 10
    if not (type(y_lines) is list):
        v_y = VanishingPoint(y_lines, img)
        right = abs(v_y[0] - w) + 10
    if not (type(z_lines) is list):
        v_z = VanishingPoint(z_lines, img)
        top = abs(v_z[0]) + 10
        bottom = abs(v_z[0]) + 10
    img = cv2pil(img)
    img = add_margin(img, top, right, bottom, left, color)
    img = pil2cv(img)
    img = cv.UMat(img)
    if not (type(x_lines) is list):
        cv.circle(img, (v_x[0] + left, v_x[1] + top), 3, (0, 0, 255), thickness=-1)
    if not (type(y_lines) is list):
        cv.circle(img, (v_y[0] + left, v_y[1] + top), 3, (0, 0, 255), thickness=-1)
    if not (type(z_lines) is list):
        cv.circle(img, (v_z[0] + left, v_z[1] + top), 3, (0, 0, 255), thickness=-1)
    img = img.get()
    print("vanishing point")
    print([v_x, v_y, v_z])
    cv.line(img, (v_x[0] + left, v_x[1] + top), (v_y[0] + left, v_y[1] + top), (0, 0, 255), 1)
    cv.circle(img, (int(w/2) + left, int(h/2) + top), 3, (0, 0, 255), thickness=-1)
    cv.imwrite('pre_result_ex01.jpg', img)

    # Calculate P-matrix  expect for 3rd column matrix
    src = [[351, 577], [191, 590], [118, 530], [408, 526]]
    src = np.array(src)
    src[:, 0] += left
    src[:, 1] += top
    p1 = []  # (1, 0, 0)
    p1.append(src[0][0])
    p1.append(src[0][1])
    p2 = []  # (0, 1, 0)
    p2.append(src[1][0])
    p2.append(src[1][1])
    p3 = []  # (0, 2, 0)
    p3.append(src[2][0])
    p3.append(src[2][1])
    p4 = []  # (2, 0, 0)
    p4.append(src[3][0])
    p4.append(src[3][1])
    _2dps = [p1, p2, p3, p4]  # Selected 4 points
    _3dps = [[1., 0], [0, 1.], [0, 2.], [2., 0]]
    H = Homography(np.array(_3dps), np.array(_2dps))  # from scene to image

    # Calculate 3rd column of P-matrix
    I = [[v_x[0] + left, v_x[1] + top, v_y[0] + left, v_y[1] + top]]
    I = np.array(I)
    I = CalcLinesEquation(I, img)
    I = I.reshape(1, 3)
    h2, w2, c = img.shape

    # Fit I equation to the image
    I[0, 2] -= (I[0, 0]*w2/2 + I[0, 1]*h2/2)
    I /= np.linalg.norm(I)
    if (type(z_lines) is list):
        v_z = np.array([[w/2 + left, 5000, 1]])
    z_alpha = CalcZ_alpha(I, -1, v_z)
    v_z *= z_alpha
    P = np.insert(H, 2, v_z, axis=1)  # P-matrix

    # 3D restruction
    img2 = cv2pil(img2)
    img2 = add_margin(img2, top, right, bottom, left, color)
    img2 = pil2cv(img2)
    x_max = 2
    y_max = 2
    z_max = 6
    scale = 0.01
    Convert3dTo2d(x_max, y_max, z_max, scale, P, img2)
    print("img2 shape")
    print(img2.shape)

    # P-matrix
    a = np.array([0, 0, 0, 1])
    b = np.array([1, 0, 0, 1])
    c = np.array([0, 1, 0, 1])
    d = np.array([0, 0, 1, 1])
    m = np.dot(P, a.T)
    m /= m[2]
    cv.circle(img, (int(m[0]), int(m[1])), 2, (0, 255, 255), thickness=-1)
    m = np.dot(P, b.T)
    m /= m[2]
    cv.circle(img, (int(m[0]), int(m[1])), 2, (0, 255, 255), thickness=-1)
    m = np.dot(P, c.T)
    m /= m[2]
    cv.circle(img, (int(m[0]), int(m[1])), 2, (0, 255, 255), thickness=-1)
    m = np.dot(P, d.T)
    m /= m[2]
    cv.circle(img, (int(m[0]), int(m[1])), 2, (0, 255, 255), thickness=-1)
    cv.imwrite('result_ex01.jpg', img)
