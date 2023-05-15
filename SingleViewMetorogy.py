import numpy as np
import cv2 as cv
from PIL import Image
import open3d as o3d


def Homography(pre_ps, tfd_ps):
    '''
    4つの対応点から射影変換を行う関数. 返値はh-matrix.
    :param pre_ps(np.array, 4*2): 変換前の４点
    :param tfd_ps(np.array, 4*2): 変換後の４点
    :return(np.array, 3*3): 射影変換行列
    '''
    assert (pre_ps.shape[0] == tfd_ps.shape[0]) & (tfd_ps.shape[0] == 4)
    h = np.ones((9, 1))
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
    h[:8] = np.dot(A_inv, tfd_ps_flatten)
    h = h.reshape((3, 3))

    return h


def SelectPoint(img):
    src_pts = []

    def onMouse1(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            src_pts.append([x, y])

    window_name1 = 'select 4 points on the ground surface. (1,0,0)->(0,1,0)->(0,2,0)->(2,0,0)'
    cv.imshow(window_name1, img)
    cv.setMouseCallback(window_name1, onMouse1)
    cv.waitKey(0)

    cv.destroyAllWindows()

    return np.array(src_pts)


def CalcLineSegmentPoints(img, horizontal_id):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 300)
    cv.imwrite('picts/edges_ex01.jpg', edges)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 130)
    lines_seg = np.zeros((lines.shape[0], 5))

    for i,line in enumerate(lines):
        rho, theta = line[0]
        # シータの値によってどの消失点に属するのかをざっくり決める
        alpha = np.pi / 18
        beta = np.pi / 18
        a = np.cos(theta)
        b = np.sin(theta)

        if (np.abs(np.cos(theta)) < np.cos(np.pi/2 - beta)):
            line_id = horizontal_id  # ここは画像により値が変わる
        elif ((np.abs(np.cos(theta)) <= np.abs(np.cos(alpha))) & (np.cos(theta) >= np.cos(np.pi/2 - beta))):
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

    cv.imwrite('picts/linesDetected_ex01.jpg', img)
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
        elif line[4] == 3:
            z_list.append(line[0:4])

    x_n = len(x_list)
    y_n = len(y_list)
    z_n = len(z_list)
    if x_n != 0:
        for i in range(x_n):
            cv.line(img, (int(x_list[i][0]), int(x_list[i][1])), (int(x_list[i][2]), int(x_list[i][3])), (255, 255, 0), 1)
        x_list = np.array(x_list).reshape(x_n, 4)
    if y_n != 0:
        for i in range(y_n):
            cv.line(img, (int(y_list[i][0]), int(y_list[i][1])), (int(y_list[i][2]), int(y_list[i][3])), (255, 0, 0), 1)
        y_list = np.array(y_list).reshape(y_n, 4)
    if z_n != 0:
        for i in range(z_n):
            cv.line(img, (int(z_list[i][0]), int(z_list[i][1])), (int(z_list[i][2]), int(z_list[i][3])), (0, 255, 255), 1)
        z_list = np.array(z_list).reshape(z_n, 4)

    return x_list, y_list, z_list


def CalcLinesEquation(lines, img):
    h, w, c = img.shape
    weight = ((h + w) / 2) / 2
    #weight = 1

    l_num = lines.shape[0]
    lines = lines.reshape(l_num, 1, 2, 2)

    e_1 = lines[:, 0, 0].astype(np.float32)
    e_2 = lines[:, 0, 1].astype(np.float32)

    # 座標変換
    e_1[:, 0] -= w / 2.
    e_2[:, 0] -= w / 2.
    e_1[:, 1] -= h / 2.
    e_2[:, 1] -= h / 2.

    # 同次座標変換
    e_1 *= weight
    e_2 *= weight
    W = np.ones((l_num, 1))*weight
    e_1 = np.append(e_1, W, axis=1)
    e_2 = np.append(e_2, W, axis=1)

    # 直線の方程式を算出
    lines_q = np.zeros((l_num, 3))

    for i in range(l_num):
        lines_q[i, :] = np.cross(e_1[i, :].reshape(1, 3), e_2[i, :].reshape(1, 3))

    return lines_q


def CalcVanishingPoints(lines):
    v_ps = []
    v_dis = []
    l_num = lines.shape[0]
    print(l_num)
    # 全ての消失点可能性を算出する
    for i in range(l_num-1):
        for j in range(i+1, l_num):

            # 平行条件
            c = lines[i, 0]*lines[j, 1] - lines[i, 1]*lines[j, 0]
            if abs(c) < 0.01:
                print('detect parallel line')
                continue
            else:
                v_ps.append((np.cross(lines[i, :], lines[j, :])).tolist())

    # 同次座標から画像座標系に変換
    v_ps = np.array(v_ps)
    v_num = len(v_ps)
    for i in range(v_num):
        v_ps[i, :] /= v_ps[i, 2]

    # 各点から全ての直線への距離の合計を算出
    for point in v_ps:
        d_sum = 0
        for line in lines:
            d_sum += CalcDistance(line, point)
        v_dis.append(d_sum)

    v_dis = np.array(v_dis)
    index = v_dis.argmin()
    v = v_ps[index, :]

    return v


def CalcDistance(line, point):  # 直線ax+by+c=0 点(x0,y0)
    numer = abs(line[0] * point[0] + line[1] * point[1] + line[2])  # 分子
    denom = np.sqrt(pow(line[0], 2) + pow(line[1], 2))  # 分母
    return numer / denom


def ReturnVanishingPoint(lines, img):
    h, w, c = img.shape
    lines_q = CalcLinesEquation(lines, img)

    v = CalcVanishingPoints(lines_q)
    v[0] += w / 2.0
    v[1] += h / 2.0

    return v.astype(np.int32)


# 画像を拡大して消失点を見せるための関数
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return result


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image

def CalcZ_alpha(v_line, Zr, V):
    src_pts = []

    def onMouse1(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            src_pts.append([x, y])

    window_name1 = 'select reference z height points. First, select a base point.'
    cv.imshow(window_name1, img)
    cv.setMouseCallback(window_name1, onMouse1)
    cv.waitKey(0)
    assert (len(src_pts) == 2)

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

def Make2dGridForRoom(w_max, h_max, depth, non_plane_num, scale, P):

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
    zeros = np.ones((1, 1)) * depth
    ones = np.ones((s, 1))

    plane = np.insert(plane, non_plane_num, zeros/scale, axis=1)
    t_plane = np.insert(t_plane, non_plane_num, zeros, axis=1)
    t_plane = np.append(t_plane, ones, axis=1)

    pts_2d = np.dot(P, t_plane.T)

    w = (pts_2d[2, :].T).reshape(s, 1)
    pts_2d = pts_2d.T / w
    pts_2d = pts_2d[:, :2]

    return pts_2d, plane


def Make2dGridForHouse03(x_max, y_max, z_max, scale):
    s_inv = int(1 / scale)
    s = int(y_max / scale)
    t = int(2.5 / scale)
    u = int(1 / scale)
    v = int(2 / scale)  # 柱の高さ / 屋根の横幅
    z = int(z_max / scale)
    # 屋根
    axis1 = np.arange(0, v, 2)
    axis1 = np.tile(axis1, s).reshape(1, s * int(v / 2))

    axis2 = np.arange(s)
    axis2 = (np.tile(axis2, int(v / 2)).reshape(int(v / 2), s)).T
    axis2 = axis2.reshape(1, s * int(v / 2))

    axis3 = np.arange(v, z).reshape(1, z - v)
    axis3 = np.tile(axis3, s).reshape(1, (z - v)*s)

    ones = np.ones(((z - v) * s, 1)) * s_inv
    plane = np.stack([axis1[0, :], axis2[0, :]], 1)
    plane = np.append(plane, axis3.T, axis=1)
    plane = np.append(plane, ones, axis=1)
    print(plane)
    xyz_1 = plane

    # 柱:高さ２，y方向に4本
    x = np.zeros((1, v))
    y = np.zeros((1, v))
    z = np.arange(v).reshape(v, 1)
    xy = np.stack([x[0, :], y[0, :]], 1)
    xyz = np.append(xy, z, axis=1)
    ones = np.ones((v, 1)) * s_inv
    xyz_2 = np.append(xyz, ones, axis=1)
    xyz_2_copy = xyz_2.copy()
    for i in range(1, 4):
        y = i * s_inv
        piller = xyz_2_copy
        piller[:, 1] = y
        #piller = np.tile(xyz_2_copy, (5, 1))
        #piller[:s_inv, 1] = y
        #piller[s_inv:s_inv * 2, 1] = y + 1
        #piller[s_inv * 2:s_inv * 3, 1] = y + 2
        #piller[s_inv * 3:s_inv * 4, 1] = y + 3
        #piller[s_inv * 4:s_inv * 5, 1] = y + 4
        xyz_2 = np.vstack([xyz_2, piller])
    xyz_1 = np.vstack([xyz_1, xyz_2])

    # 窓つきyz平面
    axis1 = np.arange(s)
    axis1 = np.tile(axis1, t).reshape(1, s*t)

    axis2 = np.arange(t)
    axis2 = (np.tile(axis2, s).reshape(s, t)).T
    axis2 = axis2.reshape(1, s*t)
    one = np.ones((1, 1)) * s_inv
    ones = np.ones((s*t, 1)) * s_inv
    plane = np.stack([axis1[0, :], axis2[0, :]], 1)
    plane = np.insert(plane, 0, one, axis=1)
    plane = np.append(plane, ones, axis=1)
    print(plane.shape)
    xyz_1 = np.vstack([xyz_1, plane])

    # 窓つきxz平面
    x = np.ones((1, t)) * s_inv
    y = np.zeros((1, t))
    z = np.arange(t).reshape(t, 1)
    xy = np.stack([x[0, :], y[0, :]], 1)
    xyz = np.append(xy, z, axis=1)
    ones = np.ones((t, 1)) * s_inv
    xyz_4 = np.append(xyz, ones, axis=1)

    for i in range(3*s_inv):
        x = 1*s_inv + i
        z_max = 0
        if (i < 1*s_inv):
            z_max = int(2.5*s_inv + i * 0.5)
        else:
            a = int(i - 1*s_inv)
            z_max = 3*s_inv - a
        x = np.ones((1, z_max)) * x
        y = np.zeros((1, z_max))
        z = np.arange(z_max).reshape(z_max, 1)
        xy = np.stack([x[0, :], y[0, :]], 1)
        xyz = np.append(xy, z, axis=1)
        ones = np.ones((z_max, 1)) * s_inv
        xyz = np.append(xyz, ones, axis=1)
        xyz_4 = np.vstack([xyz_4, xyz])
    xyz_1 = np.vstack([xyz_1, xyz_4])

    # 見えている地面
    axis1 = np.arange(u)
    axis1 = np.tile(axis1, s).reshape(1, s * u)

    axis2 = np.arange(s)
    axis2 = (np.tile(axis2, u).reshape(s, u)).T
    axis2 = axis2.reshape(1, s*u)
    zeros = np.zeros((s*u, 1))
    ones = np.ones((s*u, 1)) * s_inv
    plane = np.stack([axis1[0, :], axis2[0, :]], 1)
    plane = np.append(plane, zeros, axis=1)
    plane = np.append(plane, ones, axis=1)
    print(plane.shape)
    xyz_1 = np.vstack([xyz_1, plane])

    print(xyz_1, xyz_1.shape)

    return xyz_1


def Make3dGrid(w_max, h_max, z_max, non_plane_num, scale, P):
    '''
    :param w_max: 横
    :param h_max: 縦
    :param z_max: 奥行
    :return:
    '''
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
    '''
    バイリニア補間のための対象座標の周りの４点の座標を返す関数
    :param pts: [x, y]
    :return: [[x1, y1][x2, y2][x3, y3][x4, y4]]
    '''
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

    xz_2d, xz_3dpts = Make2dGrid(x_max, z_max, z_max, 1, scale, P)
    yz_2d, yz_3dpts = Make2dGrid(y_max, z_max, z_max, 0, scale, P)
    xy_2d, xy_3dpts = Make2dGrid(x_max, y_max, z_max, 2, scale, P)

    # 画素値を補間して置き換える
    xz_values = np.array(BilinearInterpolation(xz_2d, img))
    yz_values = np.array(BilinearInterpolation(yz_2d, img))
    xy_values = np.array(BilinearInterpolation(xy_2d, img))

    #　1つにまとめる
    _3dpts = np.vstack([xz_3dpts, yz_3dpts])
    _3dpts = np.vstack([_3dpts, xy_3dpts])
    values = np.vstack([xz_values, yz_values])
    values = np.vstack([values, xy_values])
    print(xy_3dpts)

    # 3次元データとして保存
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_3dpts)
    pcd.colors = o3d.utility.Vector3dVector(values/255.)
    o3d.estimate_normals(pcd, search_param=o3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd], "points", 1280, 960)
    o3d.io.write_point_cloud("./sync_02.ply", pcd)


def Convert3dTo2dForRoom(x_max, y_max, z_max, scale, P, img):

    xz_2d, xz_3dpts = Make2dGridForRoom(x_max, z_max, y_max, 1, scale, P)
    yz_2d, yz_3dpts = Make2dGridForRoom(y_max, z_max, x_max, 0, scale, P)
    xy_2d, xy_3dpts = Make2dGridForRoom(x_max, y_max, 0, 2, scale, P)

    # 画素値を補間して置き換える
    xz_values = np.array(BilinearInterpolation(xz_2d, img))
    yz_values = np.array(BilinearInterpolation(yz_2d, img))
    xy_values = np.array(BilinearInterpolation(xy_2d, img))

    #　1つにまとめる
    _3dpts = np.vstack([xz_3dpts, yz_3dpts])
    _3dpts = np.vstack([_3dpts, xy_3dpts])
    values = np.vstack([xz_values, yz_values])
    values = np.vstack([values, xy_values])
    print(xy_3dpts)

    # 3次元データとして保存
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_3dpts)
    pcd.colors = o3d.utility.Vector3dVector(values/255.)
    o3d.estimate_normals(pcd, search_param=o3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd], "points", 1280, 960)
    o3d.io.write_point_cloud("./sync_05.ply", pcd)


def Convert3dTo2d_ALL(x_max, y_max, z_max, scale, P, img):

    xyz = Make2dGridForHouse03(x_max, y_max, z_max, scale)
    values = np.dot(P, (xyz*scale).T)
    w = (values[2, :].T).reshape(xyz.shape[0], 1)
    values = values.T / w
    values = values[:, :2]
    # 画素値を補間して置き換える
    values = np.array(BilinearInterpolation(values, img))

    # 3次元データとして保存
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(values/255.)
    o3d.estimate_normals(pcd, search_param=o3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd], "points", 1280, 960)
    o3d.io.write_point_cloud("./sync.ply", pcd)
    #o3d.visualization.draw_geometries([pcd], "points", 1280, 960)


if __name__ == '__main__':

    # ======== Step1:画像読み込み ========
    img = cv.imread('svm5.png')
    img2 = img.copy()
    h, w, c = img.shape

    # ======== Step2:エッジ検出＆直線検出 ========
    lines = CalcLineSegmentPoints(img, 0)
    x_lines, y_lines, z_lines = DivideLines(lines, img)  # 割り当てた番号に従って直線を分類

    # ========Step3:各直線群から各方向の消失点を計算する========
    v_x, v_y, v_z = [], [], []

    top, right, bottom, left = 10, 10, 10, 10
    color = (255, 255, 255)

    if not (type(x_lines) is list):
        v_x = ReturnVanishingPoint(x_lines, img)
        left = abs(v_x[0]) + 10

    if not (type(y_lines) is list):
        v_y = ReturnVanishingPoint(y_lines, img)
        right = abs(v_y[0] - w) + 10

    if not (type(z_lines) is list):
        v_z = ReturnVanishingPoint(z_lines, img)
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

    print([v_x, v_y, v_z])
    cv.line(img, (v_x[0] + left, v_x[1] + top), (v_y[0] + left, v_y[1] + top), (0, 0, 255), 1)
    cv.circle(img, (int(w/2) + left, int(h/2) + top), 3, (0, 0, 255), thickness=-1)
    cv.imwrite('picts/pre_result_ex01.jpg', img)

    # ======== Step4:P-matrixの3列目以外を算出 ========

    # 定めた世界座標系におけるxy平面上の４点を選択し
    src = SelectPoint(img2)
    src = np.array(src)
    src[:, 0] += left
    src[:, 1] += top

    p1 = []  # (1, 0, 0)に対応する点
    p1.append(src[0][0])
    p1.append(src[0][1])

    p2 = []  # (0, 1, 0)に対応する点
    p2.append(src[1][0])
    p2.append(src[1][1])

    p3 = []  # (0, 2, 0)に対応する点
    p3.append(src[2][0])
    p3.append(src[2][1])

    p4 = []  # (2, 0, 0)に対応する点
    p4.append(src[3][0])
    p4.append(src[3][1])

    _2dps = [p1, p2, p3, p4]  # 選択した４点
    _3dps = [[1., 0], [0, 1.], [0, 2.], [2., 0]]  # それらに対応する3次元座標上の4点

    H = Homography(np.array(_3dps), np.array(_2dps))  # from scene to image

    # ======== Step5:P-matrixの3列目を計算 ========

    # Iは消失点を結んだ直線の方程式（実際には点の個数で場合分け）
    I = [[v_x[0] + left, v_x[1] + top, v_y[0] + left, v_y[1] + top]]

    I = np.array(I)
    I = CalcLinesEquation(I, img)
    I = I.reshape(1, 3)
    h2, w2, c = img.shape

    # Iの方程式を画像に合わせる
    I[0, 2] -= (I[0, 0]*w2/2 + I[0, 1]*h2/2)
    I /= np.linalg.norm(I)

    zr = -1.
    if (type(z_lines) is list):
        v_z = np.array([[w/2 + left, 5000, 1]])
    z_alpha = CalcZ_alpha(I, zr, v_z)
    v_z *= z_alpha

    P = np.insert(H, 2, v_z, axis=1)  # 透視投影行列を求める

    # ======== Step6:3次元復元 ========
    img2 = cv2pil(img2)
    img2 = add_margin(img2, top, right, bottom, left, color)
    img2 = pil2cv(img2)
    x_max = 2.5
    y_max = 4.5
    z_max = 10
    scale = 0.01
    #Convert3dTo2d_ALL(x_max, y_max, z_max, scale, P, img2)
    print(img2.shape)
    Convert3dTo2dForRoom(x_max, y_max, z_max, scale, P, img2)


    #xyz = Make2dGridForHouse03(x_max, y_max, scale)

    #pts_3d = Convert2dTo3d(img2, left, top, P_inv)
    #pcd = MakePointCloud(pts_3d, img2)
    #o3d.io.write_point_cloud("./sync.ply", pcd)

    # P-matrixの確認
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

    cv.imwrite('picts/result_ex01.jpg', img)

    '''
        minLineLength = 400
        maxLineGap = 30
        #lines = cv.HoughLinesP(edges, 0.7, np.pi / 120, 120)
        lines = cv.HoughLinesP(edges, 1, np.pi / 360, 150, minLineLength, maxLineGap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
        #lines = cv.HoughLines(edges, 0.7, np.pi / 120, 120, min_theta=np.pi / 36, max_theta=np.pi - np.pi / 36)
    '''