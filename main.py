import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
# import cvxpy as cp
import numpy as np
import pandas as pd
from mpmath import diff
import scipy
import gekko

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

muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l']

# iso = [352.62402, 625.9219, 72.08794]
# iso = [346, 628, 60]
iso = [173, 314, 30]
target_len = 50
# emg related
fs = 1000


def b_spline_basis(i, p, u, nodeVector):
    # 计算基函数，i为控制顶点序号，k为次数，u为代入的值，NodeVector为节点向量
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
        U = np.linspace(nodeVector[0], nodeVector[n + p + 1], 100)  # 在节点向量收尾之间取100个点，u在这些点中取值
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
    plt.plot(rx, ry)
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


def emg_rectification(x, Fs, code=None):
    # Fs 采样频率，在EMG信号中是1000Hz
    x_mean = np.mean(x[1:200])
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 6  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    # # yuetian
    # if code == 'BIC':
    #     # ref = 0.403
    #     # ref = 0.55
    #     ref = 0.65
    # elif code == 'BRA':
    #     # ref = 0.273
    #     ref = 0.35
    # elif code == 'BRD':
    #     # ref = 0.235
    #     ref = 0.35
    # else:
    #     ref = max(EMGLE)
    # chenzui
    # if code == 'BIC':
    #     # ref = 0.403
    #     # ref = 0.2
    #     ref = 0.65
    # elif code == 'BRA':
    #     # ref = 0.273
    #     # ref = 0.30
    #     ref = 0.26
    # elif code == 'BRD':
    #     # ref = 0.235
    #     # ref = 0.50
    #     ref = 0.8
    # yuetian
    if code == 'BIC':
        ref = 0.69173
    elif code == 'BRA':
        ref = 0.38164
    elif code == 'BRD':
        ref = 0.63951
    else:
        ref = max(EMGLE)
    # ref = 1

    normalized_EMG = EMGLE / ref
    y = normalized_EMG
    return [y, t]


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx


def resample_by_len(orig_list: list, target_len: int):
    '''
    同于标准重采样，此函数将len(list1)=x 从采样为len(list2)=y；y为指定的值，list2为输出
    :param orig_list: 是list,重采样的源列表：list1
    :param target_len: 重采样的帧数：y
    :return: 重采样后的数组:list2
    '''
    orig_list_len = len(orig_list)
    # print(orig_list_len)
    k = target_len / orig_list_len
    x = [x * k for x in range(orig_list_len)]
    x[-1] = 3572740
    if x[-1] != target_len:
        # 线性更改越界结尾
        x1 = x[-2]
        y1 = orig_list[-2]
        x2 = x[-1]
        y2 = orig_list[-1]
        y_resa = (y2 - y1) * (target_len - x1) / (x2 - x1) + y1
        x[-1] = target_len
        orig_list[-1] = y_resa
    # 使用了线性的插值方法，也可以根据需要改成其他的。推荐是线性
    f = interpolate.interp1d(x, orig_list, 'linear')
    del x
    resample_list = f([x for x in range(target_len)])
    return resample_list


def read_realted_files():
    fs = 1000
    savgol_filter_pra = 20
    emg = pd.read_excel('files/CHENYuetian_10kg.xlsx')
    so_act = pd.read_excel('files/modelModified_StaticOptimization_activation.xlsx')
    moment = pd.read_excel('files/inverse_dynamics.xlsx')
    length = pd.read_excel('files/modelModified_MuscleAnalysis_Length.xlsx')
    momarm = pd.read_excel('files/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l.xlsx')
    pforce = pd.read_excel('files/modelModified_MuscleAnalysis_PassiveFiberForce.xlsx')
    emg_mean = np.load('emg/yuetian_mean.npy')
    emg_std = np.load('emg/yuetian_std.npy')

    time_torque = moment['time']
    # time_length = length['time']
    time_momarm = momarm['time']

    timestep_emg = [0.500, 2.383, 3.533, 4.749, 5.883, 6.999, 8.316]
    t_tor = []
    t_arm = []
    tor = []
    arm = [[] for _ in range(len(muscle_idx))]

    torque = moment['elbow_flex_l_moment']
    for i in range(len(timestep_emg) - 1):
        tts = find_nearest_idx(time_torque, timestep_emg[i])
        tte = find_nearest_idx(time_torque, timestep_emg[i + 1])
        a = resample_by_len(list(time_torque[tts:tte]), target_len)
        t_tor.append(resample_by_len(list(time_torque[tts:tte]), target_len))
        tor.append(resample_by_len(list(torque[tts:tte]), target_len))

        tms = find_nearest_idx(time_momarm, timestep_emg[i])
        tme = find_nearest_idx(time_momarm, timestep_emg[i + 1])
        t_arm.append(resample_by_len(list(time_momarm[tms:tme]), target_len))
        for j in range(len(muscle_idx)):
            arm[j].append(resample_by_len(list(momarm[muscle_idx[j]][tms:tme]), target_len))

    t_tor_out = []  # 3 actions
    t_arm_out = []
    emg_mean_out = []
    emg_std_out = []
    tor_out = []
    arm_out = [[] for _ in range(len(muscle_idx))]
    for i in range(len(muscle_idx)):
        emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len * 2))
        emg_std_out.append(resample_by_len(emg_std[i, :], target_len * 2))
        for j in range(3):  # 3 actions
            arm_out[i].append(np.concatenate([arm[i][int(2 * j)], arm[i][int(2 * j + 1)]]))

    for j in range(3):  # 3 actions
        t_tor_out.append(np.concatenate([t_tor[int(2 * j)], t_tor[int(2 * j + 1)]]))
        t_arm_out.append(np.concatenate([t_arm[int(2 * j)], t_arm[int(2 * j + 1)]]))
        tor_out.append(np.concatenate([tor[int(2 * j)], tor[int(2 * j + 1)]]))

    [emg_BIC, t1] = emg_rectification(emg['BIC'], fs, 'BIC')
    [emg_BRA, t2] = emg_rectification(emg['BRA'], fs, 'BRA')
    [emg_BRD, t3] = emg_rectification(emg['BRD'], fs, 'BRD')
    emg = [([], [], []) for _ in range(len(muscle_idx))]
    time_emg = [[] for _ in range(3)]
    for i in range(3):  # 3 actions
        for t in t_tor_out[i]:
            idx = find_nearest_idx(t1, t)
            time_emg[i].append(t1[idx])
            emg[0][i].append(emg_BIC[idx])
            emg[1][i].append(emg_BRA[idx])
            emg[2][i].append(emg_BRD[idx])

    # yt = [([], [], []) for _ in range(len(muscle_idx))]
    # for i in range(len(time_yt_10) - 1):
    #     yt[j].append(resample_by_len(yt_10[j, step(time_yt_10[i]):step(time_yt_10[i + 1])], unified_len))

    # act = [[] for _ in range(len(muscle_idx))]
    # arm = [[] for _ in range(len(muscle_idx))]
    # pff = [[] for _ in range(len(muscle_idx))]
    # ml = [[] for _ in range(len(muscle_idx))]
    # for i in range(len(muscle_idx)):
    #     act[i] = scipy.signal.savgol_filter(so_act[muscle_idx[i]], savgol_filter_pra, 3)
    #     # act[i] = so_act[muscle_idx[i]]
    #     arm[i] = momarm[muscle_idx[i]]
    #     pff[i] = pforce[muscle_idx[i]]
    #     ml[i] = length[muscle_idx[i]]
    # act = np.asarray(act)
    # arm = np.asarray(arm)
    # pff = np.asarray(pff)
    # ml = np.asarray(ml)
    # torque = np.asarray(torque)

    # act[0] = so_act['bic_s_l']
    # act[1] = so_act['brachialis_1_l']
    # act[2] = so_act['brachiorad_1_l']
    # act[3] = so_act['tric_long_1_l']
    # act[3] = so_act['tric_long_2_l']
    # act[4] = so_act['tric_long_3_l']
    # act[4] = so_act['tric_long_4_l']
    # act[5] = so_act['tric_med_1_l']
    # act[5] = so_act['tric_med_2_l']
    # act[5] = so_act['tric_med_3_l']
    # act[5] = so_act['tric_med_4_l']
    # act[5] = so_act['tric_med_5_l']
    # act[6] = so_act['bic_l_l']
    # act[7] = so_act['brachiorad_2_l']
    # act[8] = so_act['brachiorad_3_l']
    # act[9] = so_act['brachialis_2_l']

    # ml = [[] for _ in range(10)]
    # ml[0] = length['bic_s_l']
    # ml[1] = length['brachialis_1_l']
    # ml[2] = length['brachiorad_1_l']
    # ml[3] = length['tric_long_1_l']
    # ml[4] = length['tric_long_3_l']
    # ml[5] = length['tric_med_1_l']
    # ml[6] = length['bic_l_l']
    # ml[7] = length['brachiorad_2_l']
    # ml[8] = length['brachiorad_3_l']
    # ml[9] = length['brachialis_2_l']
    # length1 = length['bic_s_l']
    # length2 = length['brachialis_1_l']
    # length3 = length['brachiorad_1_l']

    # arm = [[] for _ in range(10)]
    # arm[0] = momarm['bic_s_l']
    # arm[1] = momarm['brachialis_1_l']
    # arm[2] = momarm['brachiorad_1_l']
    # arm[3] = momarm['tric_long_1_l']
    # arm[4] = momarm['tric_long_3_l']
    # arm[5] = momarm['tric_med_1_l']
    # arm[6] = momarm['bic_l_l']
    # arm[7] = momarm['brachiorad_2_l']
    # arm[8] = momarm['brachiorad_3_l']
    # arm[9] = momarm['brachialis_2_l']
    # # momarm1 = momarm['bic_s_l']
    # # momarm2 = momarm['brachialis_1_l']
    # # momarm3 = momarm['brachiorad_1_l']
    # # momarm4 = momarm['bic_l_l']
    # # momarm5 = momarm['brachiorad_2_l']
    # # momarm6 = momarm['brachiorad_3_l']
    return emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg


def emg_file_progressing(emg):
    [emg_BIC, t1] = emg_rectification(emg['BIC'], fs, 'BIC')
    [emg_BRA, t2] = emg_rectification(emg['BRA'], fs, 'BRA')
    [emg_BRD, t3] = emg_rectification(emg['BRD'], fs, 'BRD')
    emg_list = np.asarray([emg_BIC, emg_BRA, emg_BRD])
    t_list = np.asarray([t1, t2, t3])
    return [emg_list, t_list]


def calculate_emg_distribution():
    unified_len = 1000
    yt_10 = pd.read_excel('emg/CHENYuetian_10kg.xlsx')
    yt_12 = pd.read_excel('emg/CHENYuetian_12kg.xlsx')
    yt_15 = pd.read_excel('emg/CHENYuetian_15kg.xlsx')

    # kh_10 = pd.read_excel('emg/kehan_10kg.xlsx')
    # kh_12 = pd.read_excel('emg/kehan_12.5kg.xlsx')
    # kh_15 = pd.read_excel('emg/kehan_15kg.xlsx')
    # cz_10 = pd.read_excel('emg/LIChenzui_10kg.xlsx')
    # cz_12 = pd.read_excel('emg/LIChenzui_12.5kg.xlsx')
    # cz_15 = pd.read_excel('emg/LIChenzui_15kg.xlsx')

    def step(time):
        return int(time * 1000 - 1)

    [yt_10, t_yt_10] = emg_file_progressing(yt_10)
    [yt_12, t_yt_12] = emg_file_progressing(yt_12)
    [yt_15, t_yt_15] = emg_file_progressing(yt_15)
    time_yt_10 = [0.500, 2.383, 3.533, 4.749, 5.883, 6.999, 8.316]
    time_yt_12 = [0.400, 1.800, 2.866, 3.783, 4.683, 5.783, 6.883]
    time_yt_15 = [0.117, 1.917, 2.917, 4.100, 5.297, 6.133, 7.117]
    yt = [([], [], []) for _ in range(3)]
    for j in range(3):  # three muscles
        for i in range(len(time_yt_10) - 1):
            yt[0][j].append(resample_by_len(yt_10[j, step(time_yt_10[i]):step(time_yt_10[i + 1])], unified_len))
            yt[1][j].append(resample_by_len(yt_12[j, step(time_yt_12[i]):step(time_yt_12[i + 1])], unified_len))
            yt[2][j].append(resample_by_len(yt_15[j, step(time_yt_15[i]):step(time_yt_15[i + 1])], unified_len))

    # yt = [yt_10, yt_12, yt_15]
    # [kh_10, t_kh_10] = emg_file_progressing(kh_10)
    # [kh_12, t_kh_12] = emg_file_progressing(kh_12)
    # [kh_15, t_kh_15] = emg_file_progressing(kh_15)
    # kh = [kh_10, kh_12, kh_15]
    # [cz_10, t_cz_10] = emg_file_progressing(cz_10)
    # [cz_12, t_cz_12] = emg_file_progressing(cz_12)
    # [cz_15, t_cz_15] = emg_file_progressing(cz_15)
    # cz = [cz_10, cz_12, cz_15]

    yt = np.asarray(yt)
    # plt.figure()
    # plt.subplot(311)
    # for i in range(3):
    #     plt.plot(yt[i, 0, 0, :])
    #     plt.plot(yt[i, 0, 2, :])
    #     plt.plot(yt[i, 0, 4, :])
    # plt.subplot(312)
    # for i in range(3):
    #     plt.plot(yt[i, 1, 0, :])
    #     plt.plot(yt[i, 1, 2, :])
    #     plt.plot(yt[i, 1, 4, :])
    # plt.subplot(313)
    # for i in range(3):
    #     plt.plot(yt[i, 2, 0, :])
    #     plt.plot(yt[i, 2, 2, :])
    #     plt.plot(yt[i, 2, 4, :])
    #
    # plt.figure()
    # plt.subplot(311)
    # for i in range(3):
    #     plt.plot(yt[i, 0, 1, :])
    #     plt.plot(yt[i, 0, 3, :])
    #     plt.plot(yt[i, 0, 5, :])
    # plt.subplot(312)
    # for i in range(3):
    #     plt.plot(yt[i, 1, 1, :])
    #     plt.plot(yt[i, 1, 3, :])
    #     plt.plot(yt[i, 1, 5, :])
    # plt.subplot(313)
    # for i in range(3):
    #     plt.plot(yt[i, 2, 1, :])
    #     plt.plot(yt[i, 2, 3, :])
    #     plt.plot(yt[i, 2, 5, :])

    plt.figure(plt.figure())
    data = [[], [], []]
    for k in range(3):  # three muscles
        for i in range(3):  # three weights
            for j in range(3):  # three actions
                data[k].append(np.concatenate([yt[i, k, int(2 * j), :], yt[i, k, int(2 * j + 1), :]]))
    data = np.asarray(data)
    data_mean = np.ones([data.shape[0], data.shape[2]])
    data_std = np.ones([data.shape[0], data.shape[2]])
    for i in range(data.shape[0]):  # three muscles
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
    color = ['#1f77b4', '#ff7f0e', '#2ca02c']
    label = ['10kg', '12.5kg', '15kg']
    plt.subplot(311)
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % 3])
    plt.ylabel('bic', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % 3])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    for i in range(data.shape[1]):
        if i < 3:
            plt.plot(data[2, i, :], color=color[i % 3], label=label[i])
        else:
            plt.plot(data[2, i, :], color=color[i % 3])
    plt.legend()
    plt.ylabel('brachiorad', weight='bold')
    plt.xlabel('timestep', weight='bold')

    plt.figure(plt.figure())
    plt.subplot(311)
    plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
    plt.ylabel('bic', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2])
    plt.ylabel('brachiorad', weight='bold')
    plt.xlabel('timestep', weight='bold')

    plt.figure(plt.figure())
    plt.subplot(311)
    plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % 3])
    plt.ylabel('bic', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % 3])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % 3])
    plt.ylabel('brachiorad', weight='bold')
    plt.xlabel('timestep', weight='bold')

    np.save('emg/yuetian_mean', data_mean)
    np.save('emg/yuetian_std', data_std)

    # print(np.max(yt[:, 0, :, :]))
    # print(np.max(yt[:, 1, :, :]))
    # print(np.max(yt[:, 2, :, :]))
    plt.show()


# if __name__ == '__main__':
#     calculate_emg_distribution()

if __name__ == '__main__':
    emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg = read_realted_files()

    # time_torque = resample_by_len(list(time_torque), 80)
    # torque = resample_by_len(list(torque), 80)
    # for i in range(3):
    #     act[i] = resample_by_len(act[i], 80)
    #     pff[i] = resample_by_len(list(pff[i]), 80)
    #     ml[i] = resample_by_len(list(ml[i]), 80)
    #     arm[i] = resample_by_len(list(arm[i]), 80)
    # emg1 = resample_by_len(emg1, 80)
    # emg2 = resample_by_len(emg2, 80)
    # emg3 = resample_by_len(emg3, 80)
    # torque_copy = torque
    # time_copy = time

    # num_start = 0
    # num_end = 398
    # num_start = 240
    # num_end = 300
    # num_start = 260
    # num_end = 290
    # num_start = 40
    # num_end = 120
    # num_start = 0
    # num_end = 300
    # num_start = 0
    # num_end = 140
    # print('start time:', time[num_start])
    # print('end time:', time[num_end])
    # dimension = 3
    iso = np.asarray(iso)
    # emg1 = np.asarray(emg1[num_start:num_end])
    # emg2 = np.asarray(emg2[num_start:num_end])
    # emg3 = np.asarray(emg3[num_start:num_end])
    arm = np.asarray(arm)
    time = np.asarray(time)
    torque = np.asarray(torque)
    emg_std = np.asarray(emg_std)
    emg_mean = np.asarray(emg_mean)
    emg = np.asarray(emg)
    time_emg = np.asarray(time_emg)

    action_idx = 1
    arm = arm[:, action_idx, :]
    time = time[action_idx, :]
    # torque = torque[action_idx, :]
    time_long = resample_by_len(list(time), 2000)

    iso_copy = [346, 628, 60]
    fr = np.asarray(
        [emg[0][action_idx] * iso_copy[0], emg[1][action_idx] * iso_copy[1], emg[2][action_idx] * iso_copy[2]])
    torque = np.array([sum(fr[:, i] * arm[:, i]) for i in range(arm.shape[1])])
    # to = torque
    # noise = np.random.normal(0, 1, to.shape)
    # torque = to + noise


    # emg = np.asarray([emg1, emg2, emg3])
    # arm = arm[:dimension, :]

    # fr = np.asarray([emg[0] * iso[0], emg[1] * iso[1], emg[2] * iso[2]])
    # torque = np.array([sum(fr[:, i] * arm[:, i]) for i in range(num_end - num_start)])
    # to = torque
    # noise = np.random.normal(0, 7, to.shape)
    # torque = to + noise
    # torque = np.asarray(torque[num_start:num_end])

    # # plt.figure()
    # plt.figure(figsize=(4, 6))
    # plt.plot(time_torque, torque, label='calculated')
    # plt.plot(time_torque, torque_copy[num_start:num_end], label='real')
    # # plt.axvline(x=time_torque[40], ymin=min(min(t), min(torque_copy[num_start:num_end])),
    # #             ymax=max(max(t), max(torque[num_start:num_end])), color='k', linestyle='--')
    # # plt.axvline(x=time_torque[120], ymin=min(min(t), min(torque_copy[num_start:num_end])),
    # #             ymax=max(max(t), max(torque[num_start:num_end])), color='k', linestyle='--')
    # # plt.axvline(x=time_torque[240], ymin=min(min(t), min(torque_copy[num_start:num_end])),
    # #             ymax=max(max(t), max(torque[num_start:num_end])), color='k', linestyle='--')
    # # plt.axvline(x=time_torque[300], ymin=min(min(t), min(torque_copy[num_start:num_end])),
    # #             ymax=max(max(t), max(torque[num_start:num_end])), color='k', linestyle='--')
    # plt.xlabel('time (s)')
    # plt.ylabel('torque')
    # plt.legend()
    # plt.show()

    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    y = m.Array(m.Var, len(muscle_idx))  # maximum isometric force
    f = m.Array(m.Var, arm.shape)  # muscle force
    t = m.Array(m.Var, torque.shape)  # torque
    # m.Minimize(np.square(x - emg).mean())
    m.Minimize(np.square(t - torque).mean())
    for i in range(arm.shape[0]):
        for j in range(arm.shape[1]):
            m.Equation(x[i][j] * y[i] == f[i, j])
            m.Equation(sum(f[:, j] * arm[:, j]) == t[j])
            m.Equation(x[i][j] <= emg_mean[i][j] + emg_std[i][j] * 2)
            m.Equation(x[i][j] >= emg_mean[i][j] - emg_std[i][j] * 2)
            # m.Equation(sum(f[:, j] * arm[:, j]) <= torque[j] * 1.1)
            # m.Equation(sum(f[:, j] * arm[:, j]) >= torque[j] * 0.9)
            # m.Equation(sum(f[:, j] * arm[:, j]) <= torque[j] + 10)
            # m.Equation(sum(f[:, j] * arm[:, j]) >= torque[j] - 10)
        # m.Equations([y[i] >= 0 * iso[i], y[i] <= 100 * iso[i]])
        m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 3 * iso[i]])
    # m.options.MAX_ITER = 100000
    m.solve(disp=False)
    # print(x)
    print(y)

    r = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i][j] = r[i][j] * x[i][j].value[0]
    e = np.ones_like(t)
    for i in range(t.shape[0]):
        e[i] = e[i] * t[i].value[0]
    t = e

    # plt.figure()
    # plt.plot(time, torque, label='measured')
    # plt.plot(time, np.asarray(t), label='optimization')
    # plt.xlabel('time (s)')
    # plt.ylabel('torque')
    # plt.legend()

    # plt.figure(figsize=(6, 6.7))
    # plt.subplot(311)
    # plt.plot(time_copy, emg1_copy, label='emg')
    # plt.plot(time, np.asarray(r[0, :]), label='optimization')
    # plt.xlabel('time (s)')
    # plt.ylabel('bic_s_l')
    # plt.legend()
    #
    # plt.subplot(312)
    # plt.plot(time_copy, emg2_copy, label='emg')
    # plt.plot(time, np.asarray(r[1, :]), label='optimization')
    # plt.xlabel('time (s)')
    # plt.ylabel('brachialis_1_l')
    # plt.legend()
    #
    # # plt.figure()
    # plt.subplot(313)
    # plt.plot(time_copy, emg3_copy, label='emg')
    # plt.plot(time, np.asarray(r[2, :]), label='optimization')
    # plt.xlabel('time (s)')
    # plt.ylabel('brachiorad_1_l')
    # plt.legend()
    # # plt.show()

    # plt.figure(figsize=(6, 6.7))
    plt.figure(figsize=(6, 7.7))
    plt.subplot(411)
    plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
    plt.plot(time, torque, label='measured', linewidth=2)
    # plt.xlabel('time (s)')
    plt.ylabel('torque', weight='bold')
    plt.legend()

    plt.subplot(412)
    plt.errorbar(time_long, emg_mean_long[0, :], 2 * emg_std_long[0, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(r[0, :]), label='optimization', linewidth=2, zorder=2)
    # plt.xlabel('time (s)')
    plt.ylabel('bic_s_l', weight='bold')
    plt.legend()

    plt.subplot(413)
    plt.errorbar(time_long, emg_mean_long[1, :], 2 * emg_std_long[1, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(r[1, :]), label='optimization', linewidth=2, zorder=2)
    # plt.xlabel('time (s)')
    plt.ylabel('brachialis_1_l', weight='bold')
    plt.legend()

    # plt.figure()
    plt.subplot(414)
    plt.errorbar(time_long, emg_mean_long[2, :], 2 * emg_std_long[2, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(r[2, :]), label='optimization', linewidth=2, zorder=2)
    plt.xlabel('time (s)', weight='bold')
    plt.ylabel('brachiorad_1_l', weight='bold')
    plt.legend()
    plt.show()
