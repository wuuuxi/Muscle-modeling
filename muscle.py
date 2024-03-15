import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import numpy as np
import pandas as pd

from require import *

# muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l', 'tric_long_1_l', 'tric_long_2_l', 'tric_long_3_l',
#               'tric_long_4_l', 'tric_med_1_l', 'tric_med_2_l', 'tric_med_3_l', 'tric_med_4_l', 'tric_med_5_l',
#               'tric_lat_1_l', 'tric_lat_2_l', 'tric_lat_3_l', 'tric_lat_4_l', 'tric_lat_5_l', 'bic_l_l',
#               'brachiorad_2_l', 'brachiorad_3_l', 'brachialis_2_l', 'brachialis_3_l', 'brachialis_4_l',
#               'brachialis_5_l', 'brachialis_6_l', 'brachialis_7_l', 'anconeus_1_l', 'anconeus_2_l', 'anconeus_3_l',
#               'anconeus_4_l', 'anconeus_5_l', 'pron_teres_1_l', 'supinator_1_l', 'supinator_2_l']
#
# iso = [352.62402, 625.9219, 72.08794, 446.0, 592.0, 566.0,
#        654, 526, 512, 480, 422, 214,
#        306, 820, 438, 550, 174, 694,
#        164, 104, 316, 314, 174,
#        202, 208, 236, 62, 140, 50,
#        236, 236, 1020, 94, 196]

# muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l']
# # muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1']
# OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404]
# MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
# PennationAngleAtOptimal = [0, 0, 0]
# KshapeActive = [0.45, 0.45, 0.45]
#
# # muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l', 'tric_long_1_l']
# # OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404, 0.09027865826431776]
# # MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
# # PennationAngleAtOptimal = [0, 0, 0, 0.174532925199]
# # KshapeActive = [0.45, 0.45, 0.45, 0.45]

all_muscles = ['bic_s_l', 'bic_l_l',
               'brachialis_1_l', 'brachialis_2_l', 'brachialis_3_l', 'brachialis_4_l', 'brachialis_5_l',
               'brachialis_6_l', 'brachialis_7_l', 'brachiorad_1_l', 'brachiorad_2_l', 'brachiorad_3_l',
               'tric_long_1_l', 'tric_long_2_l', 'tric_long_3_l', 'tric_long_4_l',
               'tric_med_1_l', 'tric_med_2_l', 'tric_med_3_l', 'tric_med_4_l', 'tric_med_5_l',
               'tric_lat_1_l', 'tric_lat_2_l', 'tric_lat_3_l', 'tric_lat_4_l', 'tric_lat_5_l',
               'anconeus_1_l', 'anconeus_2_l', 'anconeus_3_l', 'anconeus_4_l', 'anconeus_5_l',
               'pron_teres_1_l', 'supinator_1_l', 'supinator_2_l']
all_iso = [173, 347,
           314, 158, 157, 87, 101,
           104, 118, 30, 82, 52,
           223, 296, 283, 327,
           263, 256, 240, 211, 107,
           153, 410, 219, 275, 87,
           31, 70, 25, 118, 118,
           510, 47, 148]

# iso = [352.62402, 625.9219, 72.08794]
# iso = [346, 628, 60, 446.0]
# iso = [173 * 2, 314 * 7, 30 * 3]
iso = [173, 314, 30]
iso1 = [100, 100, 10]
iso2 = [6000, 6000, 3000]
# iso = [173, 314, 30, 223]
target_len = 20


def b_spline_basis(i, p, u, nodeVector):
    # 计算基函数，i为控制顶点序号，p为次数，u为代入的值，NodeVector为节点向量
    # 该函数返回第i+1个k次基函数在u处的值
    # nodeVector = np.mat(nodeVector)  # 将输入的节点转化成能够计算的数组
    # k=0时，定义一次基函数
    if p == 0:
        if (nodeVector[i] < u) & (u <= nodeVector[i + 1]):  # 若u在两个节点之间，函数之为1，否则为0
            result = 1
        else:
            result = 0
    else:
        # 计算支撑区间长度
        length1 = nodeVector[i + p] - nodeVector[i]
        length2 = nodeVector[i + p + 1] - nodeVector[i + 1]
        # 定义基函数中的两个系数
        if length1 == 0:  # 特别定义 0/0 = 0
            alpha = 0
        else:
            alpha = (u - nodeVector[i]) / length1  # alpha代表第一项
        if length2 == 0:
            beta = 0
        else:
            beta = (nodeVector[i + p + 1] - u) / length2  # beta代表第二项
        # 递归
        result = alpha * b_spline_basis(i, p - 1, u, nodeVector) + beta * b_spline_basis(i + 1, p - 1, u, nodeVector)
    return result


# 画B样条函数图像
def draw_b_spline(n, p, nodeVector, X, Y):
    plt.figure()
    basis_i = np.zeros(100)  # 存放基函数
    rx = np.zeros(100)
    ry = np.zeros(100)
    for i in range(n + 1):  # 计算第i个B样条基函数，
        # U = np.linspace(nodeVector[0], nodeVector[n + p + 1], 100)  # 在节点向量收尾之间取100个点，u在这些点中取值
        U = np.linspace(nodeVector[0], nodeVector[-1], 100)
        j = 0
        for u in U:
            nodeVector = np.array(nodeVector)
            basis_i[j] = b_spline_basis(i, p, u, nodeVector)  # 计算取u时的基函数的值
            j = j + 1
        rx = rx + X[i] * basis_i
        ry = ry + Y[i] * basis_i
        # plt.plot(U,basis_i)
        # print(basis_i)
    # print(rx)
    # print(ry)
    plt.plot(X, Y)
    plt.plot(rx[:-1], ry[:-1])
    plt.show()


def calcQuinticBezierCornerControlPoints(x0, y0, dydx0, x1, y1, dydx1, curviness):
    xC = (y1 - y0 - x1 * dydx1 + x0 * dydx0) / (dydx0 - dydx1)
    yC = (xC - x1) * dydx1 + y1
    xCx0 = (xC - x0)
    yCy0 = (yC - y0)
    xCx1 = (xC - x1)
    yCy1 = (yC - y1)
    x0x1 = (x1 - x0)
    y0y1 = (y1 - y0)

    a = xCx0 * xCx0 + yCy0 * yCy0
    b = xCx1 * xCx1 + yCy1 * yCy1
    c = x0x1 * x0x1 + y0y1 * y0y1
    assert ((c > a) & (c > b))

    x0_mid = x0 + curviness * xCx0
    y0_mid = y0 + curviness * yCy0

    x1_mid = x1 + curviness * xCx1
    y1_mid = y1 + curviness * yCy1

    xPts = [x0, x0_mid, x0_mid, x1_mid, x1_mid, x1]
    yPts = [y0, y0_mid, y0_mid, y1_mid, y1_mid, y1]
    return [xPts, yPts]


def createFiberActiveForceLengthCurve(x0, x1, x2, x3, ylow, dydx, curviness, computeIntegral, curveName):
    c = 0.1 + 0.8 * curviness
    xDelta = 0.05 * x2
    xs = (x2 - xDelta)
    y0 = 0
    dydx0 = 0

    y1 = 1 - dydx * (xs - x1)
    dydx01 = 1.25 * (y1 - y0) / (x1 - x0)

    x01 = x0 + 0.5 * (x1 - x0)
    y01 = y0 + 0.5 * (y1 - y0)
    x1s = x1 + 0.5 * (xs - x1)
    y1s = y1 + 0.5 * (1 - y1)
    dydx1s = dydx

    y2 = 1
    dydx2 = 0
    y3 = 0
    dydx3 = 0
    x23 = (x2 + xDelta) + 0.5 * (x3 - (x2 + xDelta))
    y23 = y2 + 0.5 * (y3 - y2)
    dydx23 = (y3 - y2) / ((x3 - xDelta) - (x2 + xDelta))

    p0 = calcQuinticBezierCornerControlPoints(x0, ylow, dydx0, x01, y01, dydx01, c)
    p1 = calcQuinticBezierCornerControlPoints(x01, y01, dydx01, x1s, y1s, dydx1s, c)
    p2 = calcQuinticBezierCornerControlPoints(x1s, y1s, dydx1s, x2, y2, dydx2, c)
    p3 = calcQuinticBezierCornerControlPoints(x2, y2, dydx2, x23, y23, dydx23, c)
    p4 = calcQuinticBezierCornerControlPoints(x23, y23, dydx23, x3, ylow, dydx3, c)

    ctrlPtsX = [p0[0], p1[0], p2[0], p3[0], p4[0]]
    ctrlPtsY = [p0[1], p1[1], p2[1], p3[1], p4[1]]


def calc_fiber_length(muscle_length, tendon_length, idx):
    fiber_length_at = muscle_length - tendon_length
    parallelogram_height = OptimalFiberLength[idx] * np.sin(PennationAngleAtOptimal[idx])
    maximum_sin_pennation = np.sin(MaximumPennationAngle[idx])

    if MaximumPennationAngle[idx] > 1e-9:
        minimum_fiber_length = parallelogram_height / maximum_sin_pennation
    else:
        minimum_fiber_length = OptimalFiberLength[idx] * 0.01

    minimum_fiber_length_along_tendon = minimum_fiber_length * np.cos(MaximumPennationAngle[idx])

    if fiber_length_at >= minimum_fiber_length_along_tendon:
        fiber_length = np.sqrt(parallelogram_height * parallelogram_height + fiber_length_at * fiber_length_at)
    else:
        fiber_length = minimum_fiber_length
    return fiber_length


def calc_fal(ml, tl, label):
    idx = -1
    for i in range(len(muscle_idx)):
        if muscle_idx[i] == label:
            idx = i
    if idx == -1:
        print('No suitable idx for muscle!')

    ofl = OptimalFiberLength[idx]
    ksa = KshapeActive[idx]
    lce = calc_fiber_length(muscle_length=ml, tendon_length=tl, idx=idx)
    # lce = FiberLength
    LceN = lce / ofl

    x = (LceN - 1.) * (LceN - 1.)
    fal = np.exp(-x / ksa)
    return fal


def calculate_torque():
    # emg = pd.read_excel('files/CHENYuetian_10kg.xlsx')
    so_act = pd.read_excel('files/modelModified_StaticOptimization_activation.xlsx')
    moment = pd.read_excel('files/inverse_dynamics.xlsx')
    moment_arm = pd.read_excel('files/arm/inverse_dynamics.xlsx')
    length = pd.read_excel('files/modelModified_MuscleAnalysis_Length.xlsx')
    tenlen = pd.read_excel('files/modelModified_MuscleAnalysis_TendonLength.xlsx')
    momarm = pd.read_excel('files/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l.xlsx')
    # emg_mean = np.load('emg/yuetian_mean.npy')
    # emg_std = np.load('emg/yuetian_std.npy')

    time_torque = moment['time']
    time_momarm = momarm['time']

    timestep_emg = [0.500, 2.383, 3.533, 4.749, 5.883, 6.999, 8.316]
    t_tor = []
    t_arm = []
    tor = []
    act = [[] for _ in range(len(all_muscles))]
    arm = [[] for _ in range(len(all_muscles))]
    ml = [[] for _ in range(len(all_muscles))]
    tl = [[] for _ in range(len(all_muscles))]

    torque = moment['elbow_flex_l_moment']
    torque_arm = moment_arm['elbow_flexion_moment']
    t_tor.append(list(time_torque))
    # tor.append(list(torque))

    t_arm.append(resample_by_len(list(time_momarm), target_len))
    for j in range(len(all_muscles)):
        arm[j].append(list(momarm[all_muscles[j]]))
        act[j].append(list(so_act[all_muscles[j]]))
        ml[j].append(list(length[all_muscles[j]]))
        tl[j].append(list(tenlen[all_muscles[j]]))

    arm = np.asarray(arm)
    act = np.asarray(act)
    force = []
    for i in range(len(all_muscles)):
        force.append(act[i, 0, :] * all_iso[i] * 2)
    force = np.asarray(force)
    torque_cal = np.ones_like(force[0, :])
    tor = np.ones_like(force[0, :])
    tor_arm = np.ones_like(force[0, :])
    for i in range(force.shape[1]):
        torque_cal[i] = sum(arm[:, 0, i] * force[:, i])
        tor[i] = torque[i]
        tor_arm[i] = torque_arm[i]
    plt.figure()
    plt.plot(torque_cal[:500])
    plt.plot(tor[:500])
    plt.plot(tor_arm[:500])
    plt.show()


def B_nx(n, i, x):
    if i > n:
        return 0
    elif i == 0:
        return (1-x)**n
    elif i == 1:
        return n*x*((1-x)**(n-1))
    return B_nx(n-1, i, x)*(1-x)+B_nx(n-1, i-1, x)*x


def get_value(p, canshu):
    sumx = 0.
    sumy = 0.
    length = len(p)-1
    for i in range(0, len(p)):
        sumx += (B_nx(length, i, canshu) * p[i][0])
        sumy += (B_nx(length, i, canshu) * p[i][1])
    return sumx, sumy


def get_newxy(p,x):
    xx = [0] * len(x)
    yy = [0] * len(x)
    for i in range(0, len(x)):
        # print('x[i]=', x[i])
        a, b = get_value(p, x[i])
        xx[i] = a
        yy[i] = b
        # print('xx[i]=', xx[i])
    return xx, yy


def N(i, k, T, t):  # 曲线N_ik
    if k == 0:
        if t < T[i] or t > T[i + 1]:
            return 0
        else:
            return 1
    else:
        result = (t - T[i]) / (T[i + k] - T[i]) * N(i, k - 1, T, t) + (T[i + k + 1] - t) / (
                    T[i + k + 1] - T[i + 1]) * N(i + 1, k - 1, T, t)
        return result


def main(n, V, V_num):
    T = np.linspace(0, 1, n + V_num + 1)  # T存储节点
    t_x = np.linspace(0, 1, 160)  # t_x存储每一个t值
    X = V[0]
    Y = V[1]
    x = []  # 用来存储曲线的x值
    y = []  # 用来存储曲线的y值
    for i in range(V_num - n):  # for循环用作获取第几条曲线段的数值
        result = pd.DataFrame(t_x, columns=['t'])
        for j in range(n + 1):
            result1 = []
            for t in t_x:
                result1.append(N(i + j, n, T, t))
            result['N_{0}{1}'.format(i + j, n)] = result1  # 将N_ij存入dataframe
        # 把Dataframe中 T[i+n]<=t<=T[i+n+1] 的数据取出来 保存为 matrix, 然后用matrix*np.matrix(X[i:i+n+1]).T获取曲线的x值

        # lambda x: x>T[i+n] and x<T[i+n+1] 很奇怪，用 >= 或 <= 曲线有时会有问题 比如 V = [[0,0,2,3,2,1],[1,3,3,2,1,1]]
        Ni_matrix = np.matrix(result[result['t'].apply(lambda x: x >= T[i + n] and x <= T[i + n + 1])].iloc[:,
                              1:])  # Ni_matrix 是一个 t_ba*j 维矩阵

        x = x + ((Ni_matrix * np.matrix(X[i:i + n + 1]).T).T).tolist()[0]
        y = y + ((Ni_matrix * np.matrix(Y[i:i + n + 1]).T).T).tolist()[0]

    fig = plt.figure()
    plt.plot(X, Y, marker='o', markerfacecolor='white')
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # n = int(input('请输入曲线的次数：'))
    # V_num = int(input('请输入控制顶点个数：'))
    n = 2
    V_num = 6
    V = [[1, 1.5, 3, 3.2, 5, 8, 8], [0, 3, 5, -0.3, -0.8, 2, 2]]
    # V = [[1, 1, 1.5, 3, 3.8, 3.2, 5, 8, 8], [0, 0, 2.6, 3, 2.2, -0.3, -0.8, 2, 2]]  # 有重复型值点坐标
    main(n, V, V_num)

#
# if __name__ == '__main__':
#     # calculate_torque()
#     # print(calc_fal(0.1, 'bic_s_l'))
#     # plt.show()
#
#     n = 5
#     p = 3
#
#     # nodeVector = [0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 0.9, 1, 1, 1, 1]  # 节点向量
#     nodeVector = np.linspace(0, 2, n+p+1)  # 节点向量
#
#     X = [0, 1, 2, 3, 4, 5, 6]
#     Y = [0, 3, 1, 3, 1, 4, 1]
#     draw_b_spline(n, p, nodeVector, X, Y)
#
#     # p = np.array([  # 控制点，控制贝塞尔曲线的阶数n
#     #     [2, -4],
#     #     [3, 8],
#     #     [5, 1],
#     #     [7, 6],
#     #     [9, 4],
#     #     [7, 1],
#     #     [8, 5],
#     # ])
#     #
#     # x = np.linspace(0, 1, 101)
#     # xx, yy = get_newxy(p, x)
#     # plt.plot(xx, yy, 'r', linewidth=1)  # 最终拟合的贝塞尔曲线
#     # plt.scatter(xx[:], yy[:], 1, "blue")  # 散点图,表示采样点
#     # plt.show()
#
