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
# muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1']
OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404]
MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
PennationAngleAtOptimal = [0, 0, 0]
KshapeActive = [0.45, 0.45, 0.45]

# muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l', 'tric_long_1_l']
# OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404, 0.09027865826431776]
# MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
# PennationAngleAtOptimal = [0, 0, 0, 0.174532925199]
# KshapeActive = [0.45, 0.45, 0.45, 0.45]

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
emg_lift_len = 1000
target_len = 20
# emg related
# fs = 2000
only_lift = False
mvc_is_variable = False
plot_distribution = True
test_plot_distribution = False
include_TRI = False
torque_init_0 = False


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


def emg_rectification(x, Fs=1000, code=None, people=None):
    # Fs 采样频率，在EMG信号中是1000Hz
    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 6  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    # if code == 'BIC':
    #     ref = 63.4452
    # elif code == 'BRA':
    #     ref = 29.9301
    # elif code == 'BRD':
    #     ref = 116.8316
    # elif code == 'TRI':
    #     ref = 42.5301
    # else:
    #     ref = max(EMGLE)
    if people == 'chenzui':
        if code == 'BIC':
            ref = 407.11
        elif code == 'BRA':
            ref = 250.37
        elif code == 'BRD':
            ref = 468.31
        elif code == 'TRI':
            ref = 467.59
        else:
            ref = max(EMGLE)
    elif people == 'zhuo':
        if code == 'BIC':
            ref = 417.90
        elif code == 'BRA':
            ref = 126.73
        elif code == 'BRD':
            ref = 408.73
        elif code == 'TRI':
            ref = 250.28
        else:
            ref = max(EMGLE)
    else:
        print('No information of this people.')
    # ref = 1

    normalized_EMG = EMGLE / ref
    y = normalized_EMG
    return [y, t]


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx


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


def read_realted_files(label='chenzui-left-3kg', idx='1'):
    '''
    :param label:
    :param idx:
    :return: emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    '''
    fs = 2000
    # if label == '11':
    #     emg = np.load('files/chenzui-3kg/emg-11.npy')
    #     moment = pd.read_excel('files/chenzui-3kg/inverse_dynamics-11.xlsx')
    #     length = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_Length-11.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_TendonLength-11.xlsx')
    #     momarm = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-11.xlsx')
    # elif label == '13':
    #     emg = np.load('files/chenzui-3kg/emg-13.npy')
    #     moment = pd.read_excel('files/chenzui-3kg/inverse_dynamics-13.xlsx')
    #     length = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_Length-13.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_TendonLength-13.xlsx')
    #     momarm = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-13.xlsx')
    # elif label == '14':
    #     emg = np.load('files/chenzui-3kg/emg-14.npy')
    #     moment = pd.read_excel('files/chenzui-3kg/inverse_dynamics-14.xlsx')
    #     length = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_Length-14.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_TendonLength-14.xlsx')
    #     momarm = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-14.xlsx')
    # elif label == '15':
    #     emg = np.load('files/chenzui-3kg/emg-15.npy')
    #     moment = pd.read_excel('files/chenzui-3kg/inverse_dynamics-15.xlsx')
    #     length = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_Length-15.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_TendonLength-15.xlsx')
    #     momarm = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-15.xlsx')
    # elif label == '17':
    #     emg = np.load('files/chenzui-3kg/emg-17.npy')
    #     moment = pd.read_excel('files/chenzui-3kg/inverse_dynamics-17.xlsx')
    #     length = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_Length-17.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_TendonLength-17.xlsx')
    #     momarm = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-17.xlsx')
    # elif label == '18':
    #     emg = np.load('files/chenzui-3kg/emg-18.npy')
    #     moment = pd.read_excel('files/chenzui-3kg/inverse_dynamics-18.xlsx')
    #     length = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_Length-18.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_TendonLength-18.xlsx')
    #     momarm = pd.read_excel('files/chenzui-3kg/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-18.xlsx')
    # elif label == '5.5kg-2':
    #     emg = np.load('files/chenzui-5.5kg/2.npy')
    #     moment = pd.read_excel('files/chenzui-5.5kg/inverse_dynamics-2.xlsx')
    #     length = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-2.xlsx')
    #     momarm = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
    # elif label == '5.5kg-3':
    #     emg = np.load('files/chenzui-5.5kg/3.npy')
    #     moment = pd.read_excel('files/chenzui-5.5kg/inverse_dynamics-3.xlsx')
    #     length = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-3.xlsx')
    #     momarm = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
    # elif label == '5.5kg-4':
    #     emg = np.load('files/chenzui-5.5kg/4.npy')
    #     moment = pd.read_excel('files/chenzui-5.5kg/inverse_dynamics-4.xlsx')
    #     length = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_Length-4.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-4.xlsx')
    #     momarm = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_TendonLength-4.xlsx')
    # elif label == '5.5kg-5':
    #     emg = np.load('files/chenzui-5.5kg/5.npy')
    #     moment = pd.read_excel('files/chenzui-5.5kg/inverse_dynamics-5.xlsx')
    #     length = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_Length-5.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-5.xlsx')
    #     momarm = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_TendonLength-5.xlsx')
    # elif label == '5.5kg-6':
    #     emg = np.load('files/chenzui-5.5kg/6.npy')
    #     moment = pd.read_excel('files/chenzui-5.5kg/inverse_dynamics-6.xlsx')
    #     length = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-6.xlsx')
    #     momarm = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
    # elif label == '5.5kg-7':
    #     emg = np.load('files/chenzui-5.5kg/7.npy')
    #     moment = pd.read_excel('files/chenzui-5.5kg/inverse_dynamics-7.xlsx')
    #     length = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
    #     tenlen = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-7.xlsx')
    #     momarm = pd.read_excel('files/chenzui-5.5kg/Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
    if label == 'chenzui-left-3kg':
        if idx == '11':
            emg = np.load('files/chenzui-3kg-1/emg-11.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-11.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-11.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-11.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-11.xlsx')
        elif idx == '13':
            emg = np.load('files/chenzui-3kg-1/emg-13.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-13.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-13.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-13.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-13.xlsx')
        elif idx == '14':
            emg = np.load('files/chenzui-3kg-1/emg-14.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-14.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-14.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-14.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-14.xlsx')
        elif idx == '15':
            emg = np.load('files/chenzui-3kg-1/emg-15.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-15.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-15.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-15.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-15.xlsx')
        elif idx == '17':
            emg = np.load('files/chenzui-3kg-1/emg-17.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-17.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-17.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-17.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-17.xlsx')
        elif idx == '18':
            emg = np.load('files/chenzui-3kg-1/emg-18.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-18.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-18.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-18.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-18.xlsx')
        elif idx == '19':
            emg = np.load('files/chenzui-3kg-1/emg-19.npy')
            moment = pd.read_excel('files/chenzui-3kg-1/inverse_dynamics-19.xlsx')
            length = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_Length-19.xlsx')
            tenlen = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_TendonLength-19.xlsx')
            momarm = pd.read_excel('files/chenzui-3kg-1/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-19.xlsx')
        else:
            print('No such index!')
    elif label == 'chenzui-left-5.5kg':
        if idx == '2':
            emg = np.load('files/chenzui-5.5kg-1/2.npy')
            moment = pd.read_excel('files/chenzui-5.5kg-1/inverse_dynamics-2.xlsx')
            length = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
            tenlen = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
            momarm = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-2.xlsx')
        elif idx == '3':
            emg = np.load('files/chenzui-5.5kg-1/3.npy')
            moment = pd.read_excel('files/chenzui-5.5kg-1/inverse_dynamics-3.xlsx')
            length = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
            tenlen = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
            momarm = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-3.xlsx')
        elif idx == '4':
            emg = np.load('files/chenzui-5.5kg-1/4.npy')
            moment = pd.read_excel('files/chenzui-5.5kg-1/inverse_dynamics-4.xlsx')
            length = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_Length-4.xlsx')
            tenlen = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_TendonLength-4.xlsx')
            momarm = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-4.xlsx')
        elif idx == '5':
            emg = np.load('files/chenzui-5.5kg-1/5.npy')
            moment = pd.read_excel('files/chenzui-5.5kg-1/inverse_dynamics-5.xlsx')
            length = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_Length-5.xlsx')
            tenlen = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_TendonLength-5.xlsx')
            momarm = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-5.xlsx')
        elif idx == '6':
            emg = np.load('files/chenzui-5.5kg-1/6.npy')
            moment = pd.read_excel('files/chenzui-5.5kg-1/inverse_dynamics-6.xlsx')
            length = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
            tenlen = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
            momarm = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-6.xlsx')
        elif idx == '7':
            emg = np.load('files/chenzui-5.5kg-1/7.npy')
            moment = pd.read_excel('files/chenzui-5.5kg-1/inverse_dynamics-7.xlsx')
            length = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
            tenlen = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
            momarm = pd.read_excel('files/chenzui-5.5kg-1/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-7.xlsx')
        else:
            print('No such index!')
    elif label == 'zhuo-left-1kg':
        if idx == '2':
            emg = np.load('files/lizhuo-1kg/2.npy')
            moment = pd.read_excel('files/lizhuo-1kg/inverse_dynamics-2.xlsx')
            length = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
            tenlen = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
            momarm = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-2.xlsx')
        elif idx == '3':
            emg = np.load('files/lizhuo-1kg/3.npy')
            moment = pd.read_excel('files/lizhuo-1kg/inverse_dynamics-3.xlsx')
            length = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
            tenlen = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
            momarm = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-3.xlsx')
        elif idx == '4':
            emg = np.load('files/lizhuo-1kg/4.npy')
            moment = pd.read_excel('files/lizhuo-1kg/inverse_dynamics-4.xlsx')
            length = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_Length-4.xlsx')
            tenlen = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_TendonLength-4.xlsx')
            momarm = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-4.xlsx')
        elif idx == '6':
            emg = np.load('files/lizhuo-1kg/6.npy')
            moment = pd.read_excel('files/lizhuo-1kg/inverse_dynamics-6.xlsx')
            length = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
            tenlen = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
            momarm = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-6.xlsx')
        elif idx == '7':
            emg = np.load('files/lizhuo-1kg/7.npy')
            moment = pd.read_excel('files/lizhuo-1kg/inverse_dynamics-7.xlsx')
            length = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
            tenlen = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
            momarm = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-7.xlsx')
        elif idx == '9':
            emg = np.load('files/lizhuo-1kg/9.npy')
            moment = pd.read_excel('files/lizhuo-1kg/inverse_dynamics-9.xlsx')
            length = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_Length-9.xlsx')
            tenlen = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_TendonLength-9.xlsx')
            momarm = pd.read_excel('files/lizhuo-1kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-9.xlsx')
        else:
            print('No such index!')
    elif label == 'zhuo-right-3kg':
        if idx == '1':
            emg = np.load('files/lizhuo-3kg/1.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-1.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-1.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-1.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-1.xlsx')
        elif idx == '2':
            emg = np.load('files/lizhuo-3kg/2.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-2.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-2.xlsx')
        elif idx == '3':
            emg = np.load('files/lizhuo-3kg/3.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-3.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-3.xlsx')
        elif idx == '5':
            emg = np.load('files/lizhuo-3kg/5.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-5.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-5.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-5.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-5.xlsx')
        elif idx == '6':
            emg = np.load('files/lizhuo-3kg/6.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-6.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-6.xlsx')
        elif idx == '7':
            emg = np.load('files/lizhuo-3kg/7.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-7.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-7.xlsx')
        elif idx == '8':
            emg = np.load('files/lizhuo-3kg/8.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-8.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-8.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-8.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-8.xlsx')
        elif idx == '9':
            emg = np.load('files/lizhuo-3kg/9.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-9.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-9.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-9.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-9.xlsx')
        elif idx == '10':
            emg = np.load('files/lizhuo-3kg/10.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-10.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-10.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-10.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-10.xlsx')
        elif idx == '11':
            emg = np.load('files/lizhuo-3kg/10.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-11.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-11.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-11.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-11.xlsx')
        elif idx == '12-1' or idx == '12-2' or idx == '12-3':
            emg = np.load('files/lizhuo-3kg/12.npy')
            moment = pd.read_excel('files/lizhuo-3kg/inverse_dynamics-12.xlsx')
            length = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_Length-12.xlsx')
            tenlen = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_TendonLength-12.xlsx')
            momarm = pd.read_excel('files/lizhuo-3kg/Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-12.xlsx')
    elif label == 'chenzui-left-all-2.5kg':
        if idx == '5':
            emg = np.load('files/chenzui-all/2.5-5.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-5.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-5.xlsx')
        elif idx == '6':
            emg = np.load('files/chenzui-all/2.5-6.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-6.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-6.xlsx')
        elif idx == '7':
            emg = np.load('files/chenzui-all/2.5-7.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-7.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-7.xlsx')
        elif idx == '8':
            emg = np.load('files/chenzui-all/2.5-8.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-8.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-8.xlsx')
        elif idx == '9':
            emg = np.load('files/chenzui-all/2.5-9.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-9.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-9.xlsx')
        else:
            print('No such index!')
    elif label == 'chenzui-left-all-3kg':
        if idx == '10':
            emg = np.load('files/chenzui-all/3-10.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-10.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-10.xlsx')
        elif idx == '11':
            emg = np.load('files/chenzui-all/3-11.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-11.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-11.xlsx')
        elif idx == '12':
            emg = np.load('files/chenzui-all/3-12.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-12.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-12.xlsx')
        elif idx == '13':
            emg = np.load('files/chenzui-all/3-13.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-13.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-13.xlsx')
        elif idx == '14':
            emg = np.load('files/chenzui-all/3-14.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-14.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-14.xlsx')
        else:
            print('No such index!')
    elif label == 'chenzui-left-all-4kg':
        if idx == '15':
            emg = np.load('files/chenzui-all/4-15.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-15.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-15.xlsx')
        elif idx == '16':
            emg = np.load('files/chenzui-all/4-16.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-16.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-16.xlsx')
        elif idx == '17':
            emg = np.load('files/chenzui-all/4-17.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-17.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-17.xlsx')
        else:
            print('No such index!')
    elif label == 'chenzui-left-all-6.5kg':
        if idx == '18':
            emg = np.load('files/chenzui-all/6.5-18.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-18.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-18.xlsx')
        elif idx == '19':
            emg = np.load('files/chenzui-all/6.5-19.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-19.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-19.xlsx')
        elif idx == '20':
            emg = np.load('files/chenzui-all/6.5-20.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-20.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-20.xlsx')
        elif idx == '21':
            emg = np.load('files/chenzui-all/6.5-21.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-21.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-21.xlsx')
        elif idx == '22':
            emg = np.load('files/chenzui-all/6.5-22.npy')
            moment = pd.read_excel('files/chenzui-all/inverse_dynamics-22.xlsx')
            momarm = pd.read_excel('files/chenzui-all/modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-22.xlsx')
        else:
            print('No such index!')
    else:
        print('No corresponding label!')
    # momarm = tenlen

    if label == 'chenzui-left-3kg':
        emg_mean = np.load('emg/chenzui_mean_3kg.npy')
        emg_std = np.load('emg/chenzui_std_3kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_3kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_3kg.npy')
    elif label == 'chenzui-left-5.5kg':
        emg_mean = np.load('emg/chenzui_mean_5.5kg.npy')
        emg_std = np.load('emg/chenzui_std_5.5kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_5.5kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_5.5kg.npy')
    elif label == 'zhuo-left-1kg':
        emg_mean = np.load('emg/lizhuo_mean_1kg.npy')
        emg_std = np.load('emg/lizhuo_std_1kg.npy')
        emg_trend_u = np.load('emg/lizhuo_trend_u_1kg.npy')
        emg_trend_d = np.load('emg/lizhuo_trend_d_1kg.npy')
    elif label == 'zhuo-right-3kg':
        emg_mean = np.load('emg/lizhuo_mean_3kg.npy')
        emg_std = np.load('emg/lizhuo_std_3kg.npy')
        emg_trend_u = np.load('emg/lizhuo_trend_u_3kg.npy')
        emg_trend_d = np.load('emg/lizhuo_trend_d_3kg.npy')
    elif label == 'chenzui-left-all-2.5kg':
        emg_mean = np.load('emg/chenzui_mean_2.5kg.npy')
        emg_std = np.load('emg/chenzui_std_2.5kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_2.5kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_2.5kg.npy')
    elif label == 'chenzui-left-all-3kg':
        emg_mean = np.load('emg/chenzui_mean_3kg.npy')
        emg_std = np.load('emg/chenzui_std_3kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_3kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_3kg.npy')
    elif label == 'chenzui-left-all-4kg':
        emg_mean = np.load('emg/chenzui_mean_4kg.npy')
        emg_std = np.load('emg/chenzui_std_4kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_4kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_4kg.npy')
    elif label == 'chenzui-left-all-6.5kg':
        emg_mean = np.load('emg/chenzui_mean_6.5kg.npy')
        emg_std = np.load('emg/chenzui_std_6.5kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_6.5kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_6.5kg.npy')
    # elif label == 'chenzui-left-all-2.5kg' or label == 'chenzui-left-all-3kg' or label == 'chenzui-left-all-4kg' or label == 'chenzui-left-all-6.5kg':
    #     emg_mean = np.load('emg/chenzui_mean_all.npy')
    #     emg_std = np.load('emg/chenzui_std_all.npy')
    #     emg_trend_u = np.load('emg/chenzui_trend_u_all.npy')
    #     emg_trend_d = np.load('emg/chenzui_trend_d_all.npy')

    time_torque = moment['time']
    time_momarm = momarm['time']

    if label == 'chenzui-left-3kg':
        if idx == '11':
            timestep_emg = [50.398, 57.531, 57.931, 70.13]
        elif idx == '13':
            timestep_emg = [214.69, 223.09, 223.09, 236.139]
        elif idx == '14':
            timestep_emg = [295.103, 303.336, 303.686, 316.469]
        elif idx == '15':
            timestep_emg = [395.348, 402.648, 403.398, 415.981]
        elif idx == '17':
            timestep_emg = [568.907, 577.106, 577.106, 590.106]
        elif idx == '18':
            timestep_emg = [644.553, 653.669, 653.669, 668.352]
    elif label == 'chenzui-left-5.5kg':
        if idx == '2':
            timestep_emg = [17.582, 28.415, 28.415, 43.481]
        elif idx == '3':
            timestep_emg = [17.033, 27.482, 27.482, 42.531]
        elif idx == '4':
            timestep_emg = [19.216, 29.799, 30.132, 44.398]
        elif idx == '5':
            timestep_emg = [19.382, 30.648, 30.648, 46.048]
        elif idx == '6':
            timestep_emg = [18.149, 28.249, 28.332, 42.515]
        elif idx == '7':
            timestep_emg = [20.15, 30.299, 30.932, 44.298]
    elif label == 'zhuo-left-1kg':
        if idx == '2':
            timestep_emg = [20.016, 27.599, 27.599, 37.465]
        elif idx == '3':
            timestep_emg = [14.365, 25.798, 25.798, 36.248]
        elif idx == '4':
            timestep_emg = [12.766, 27.032, 27.032, 39.865]
        elif idx == '6':
            timestep_emg = [13.483, 25.832, 25.866, 55.331]
        elif idx == '7':
            timestep_emg = [12.566, 29.299, 29.299, 53.014]
        elif idx == '9':
            timestep_emg = [12.199, 29.698, 29.698, 51.514]
    elif label == 'zhuo-right-3kg':
        if idx == '1':
            timestep_emg = [20.383, 32.832, 32.832, 46.348]
        elif idx == '2':
            timestep_emg = [19.250, 32.449, 32.449, 47.765]
        elif idx == '3':
            timestep_emg = [14.866, 29.916, 29.916, 48.015]
        elif idx == '5':
            timestep_emg = [12.099, 24.215, 25.932, 42.098]
        elif idx == '6':
            timestep_emg = [14.133, 30.015, 30.015, 50.098]
        elif idx == '7':
            timestep_emg = [15.933, 24.233, 24.233, 37.882]
        elif idx == '8':
            timestep_emg = [15.332, 25.449, 25.449, 39.265]
        elif idx == '9':
            timestep_emg = [13.7, 27.116, 27.116, 37.649]
        elif idx == '10':
            timestep_emg = [12.8, 20.0, 20.0, 28.799]  # 7.20s, 8.80s
        elif idx == '11':
            timestep_emg = [16.099, 25.332, 25.332, 37.848]
        elif idx == '12-1':
            timestep_emg = [19.032, 21.649, 21.649, 24.465]  # 2.62s, 2.82s
        elif idx == '12-2':
            timestep_emg = [24.465, 26.865, 26.865, 30.165]  # 2.40s, 3.30s
        elif idx == '12-3':
            timestep_emg = [30.665, 33.198, 33.198, 36.498]  # 2.53s, 3.30s
    elif label == 'chenzui-left-all-2.5kg':
        if idx == '5':
            timestep_emg = [16.416, 27.065, 27.065, 41.681]
        elif idx == '6':
            timestep_emg = [13.750, 24.782, 24.782, 42.731]
        elif idx == '7':
            timestep_emg = [13.649, 25.982, 25.982, 41.598]
        elif idx == '8':
            timestep_emg = [15.832, 28.482, 28.482, 44.864]
        elif idx == '9':
            timestep_emg = [15.549, 26.365, 26.599, 40.031]
    elif label == 'chenzui-left-all-3kg':
        if idx == '10':
            timestep_emg = [15.599, 26.249, 26.532, 39.348]
        elif idx == '11':
            timestep_emg = [15.216, 25.582, 25.582, 40.781]
        elif idx == '12':
            timestep_emg = [16.066, 28.299, 28.299, 40.882]
        elif idx == '13':
            timestep_emg = [12.049, 22.699, 22.699, 34.965]
        elif idx == '14':
            timestep_emg = [17.849, 29.365, 29.365, 43.381]
    elif label == 'chenzui-left-all-4kg':
        if idx == '15':
            timestep_emg = [14.966, 24.216, 24.216, 37.399]
        elif idx == '16':
            timestep_emg = [13.133, 23.899, 23.899, 37.565]
        elif idx == '17':
            timestep_emg = [11.433, 22.516, 22.516, 35.682]
    elif label == 'chenzui-left-all-6.5kg':
        if idx == '18':
            timestep_emg = [11.233, 23.465, 23.465, 37.948]
        elif idx == '19':
            timestep_emg = [11.833, 25.132, 25.132, 36.548]
        elif idx == '20':
            timestep_emg = [14.633, 27.399, 27.399, 40.465]
        elif idx == '21':
            timestep_emg = [11.9, 24.599, 24.599, 37.465]
        elif idx == '22':
            timestep_emg = [19.466, 30.415, 31.749, 45.781]

    t_tor = []
    t_arm = []
    tor = []
    arm = [[] for _ in range(len(muscle_idx))]
    ml = [[] for _ in range(len(muscle_idx))]
    tl = [[] for _ in range(len(muscle_idx))]
    fa = [[] for _ in range(len(muscle_idx))]
    fr = []

    if label == 'zhuo-right-3kg':
        torque = moment['elbow_flex_r_moment']
    else:
        torque = moment['elbow_flex_l_moment']

    for i in range(int(len(timestep_emg) / 2)):
        tts = find_nearest_idx(time_torque, timestep_emg[2 * i])
        tte = find_nearest_idx(time_torque, timestep_emg[2 * i + 1])
        t_tor.append(resample_by_len(list(time_torque[tts:tte]), target_len))
        tor.append(resample_by_len(list(torque[tts:tte]), target_len))

        tms = find_nearest_idx(time_momarm, timestep_emg[2 * i])
        tme = find_nearest_idx(time_momarm, timestep_emg[2 * i + 1])
        t_arm.append(resample_by_len(list(time_momarm[tms:tme]), target_len))
        for j in range(len(muscle_idx)):
            arm[j].append(resample_by_len(list(momarm[muscle_idx[j]][tms:tme]), target_len))
            # ml[j].append(resample_by_len(list(length[muscle_idx[j]][tms:tme]), target_len))
            # tl[j].append(resample_by_len(list(tenlen[muscle_idx[j]][tms:tme]), target_len))
            # for k in range(tms, tme):
            #     fr.append(calc_fal(length[muscle_idx[j]][k], tenlen[muscle_idx[j]][k], muscle_idx[j]))
            # fa[j].append(resample_by_len(fr, target_len))
            fr = []

    t_tor_out = []  # 3 actions
    t_arm_out = []
    emg_mean_out = []
    emg_std_out = []
    emg_trend_u_out = []
    emg_trend_d_out = []
    tor_out = []
    arm_out = [[] for _ in range(len(muscle_idx))]
    ml_out = [[] for _ in range(len(muscle_idx))]
    tl_out = [[] for _ in range(len(muscle_idx))]
    fa_out = [[] for _ in range(len(muscle_idx))]
    for i in range(len(muscle_idx)):
        if only_lift is False:
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :], target_len * 2))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :], target_len * 2))
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len * 2))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len * 2))
            arm_out[i].append(np.concatenate([arm[i][0], arm[i][1]]))
            # ml_out[i].append(np.concatenate([ml[i][0], ml[i][1]]))
            # tl_out[i].append(np.concatenate([tl[i][0], tl[i][1]]))
            # fa_out[i].append(np.concatenate([fa[i][0], fa[i][1]]))
        else:
            emg_mean = emg_mean[:, :emg_lift_len]
            emg_std = emg_std[:, :emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len))
            arm_out[i].append(arm[i][0])
            # ml_out[i].append(ml[i][0])
            # tl_out[i].append(tl[i][0])
            # fa_out[i].append(fa[i][0])

    if only_lift is False:
        t_tor_out.append(np.concatenate([t_tor[0], t_tor[1]]))
        t_arm_out.append(np.concatenate([t_arm[0], t_arm[1]]))
        tor_out.append(np.concatenate([tor[0], tor[1]]))
    else:
        t_tor_out.append(t_tor[0])
        t_arm_out.append(t_arm[0])
        tor_out.append(tor[0])

    if label == 'chenzui-left-3kg' or label == 'chenzui-left-5.5kg' or label == 'chenzui-left-all-2.5kg' or label == 'chenzui-left-all-3kg' or label == 'chenzui-left-all-4kg' or label == 'chenzui-left-all-6.5kg':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', 'chenzui')
        [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', 'chenzui')
        [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', 'chenzui')
        if include_TRI is True:
            [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', 'chenzui')
    elif label == 'zhuo-left-1kg' or label == 'zhuo-right-3kg':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', 'zhuo')
        [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', 'zhuo')
        [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', 'zhuo')
        if include_TRI is True:
            [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', 'zhuo')

    # [emg_TRIlat, t5] = emg_rectification(emg['TRIlateral'], fs, 'TRIlat')
    if label == 'chenzui-left-3kg':
        if idx == '11':
            t1 = t1 - 4.131 + 45.182  # 11
        elif idx == '13':
            t1 = t1 - 4.763 + 209.74  # 13
        elif idx == '14':
            t1 = t1 - 7.0005 + 289.653  # 14
        elif idx == '15':
            t1 = t1 - 14.9555 + 391.032  # 15
        elif idx == '17':
            t1 = t1 - 4.426 + 563.707  # 17
        elif idx == '18':
            t1 = t1 - 4.03 + 638.553  # 18
    elif label == 'chenzui-left-5.5kg':
        if idx == '2':
            t1 = t1 - 3.638 + 12.099
        elif idx == '3':
            t1 = t1 - 3.0785 + 11.783
        elif idx == '4':
            t1 = t1 - 4.427 + 12.933
        elif idx == '5':
            t1 = t1 - 5.5625 + 14.066
        elif idx == '6':
            t1 = t1 - 5.274 + 11.65
        elif idx == '7':
            t1 = t1 - 3.169 + 13.967
    elif label == 'zhuo-left-1kg':
        if idx == '2':
            t1 = t1 - 11.3095 + 16.0
        elif idx == '3':
            t1 = t1 - 4.0 + 12.149
        elif idx == '4':
            t1 = t1 - 4.3385 + 11.299
        elif idx == '6':
            t1 = t1 - 4.9365 + 11.283
        elif idx == '7':
            t1 = t1 - 4.8945 + 11.033
        elif idx == '9':
            t1 = t1 - 4.2645 + 9.899
    elif label == 'zhuo-right-3kg':
        if idx == '1':
            t1 = t1 - 4.2525 + 13.067
        elif idx == '2':
            t1 = t1 - 8.14 + 14.1
        elif idx == '3':
            t1 = t1 - 3.6615 + 9.866
        elif idx == '5':
            t1 = t1 - 3.642 + 7.749
        elif idx == '6':
            t1 = t1 - 4.103 + 9.716
        elif idx == '7':
            t1 = t1 - 7.3255 + 13.183
        elif idx == '8':
            t1 = t1 - 3.873 + 9.866
        elif idx == '9':
            t1 = t1 - 4.2335 + 9.75
        elif idx == '10':
            t1 = t1 - 4.1045 + 9.367
        elif idx == '11':
            t1 = t1 - 5.6695 + 12.766
        elif idx == '12-1':
            t1 = t1 - 7.29 + 14.749
        elif idx == '12-2':
            t1 = t1 - 7.29 + 14.749
        elif idx == '12-3':
            t1 = t1 - 7.29 + 14.749
    elif label == 'chenzui-left-all-2.5kg':
        if idx == '5':
            t1 = t1 - 3.464 + 9.733
        elif idx == '6':
            t1 = t1 - 3.7925 + 7.65
        elif idx == '7':
            t1 = t1 - 3.2645 + 8.583
        elif idx == '8':
            t1 = t1 - 3.178 + 11.249
        elif idx == '9':
            t1 = t1 - 4.6665 + 9.466
    elif label == 'chenzui-left-all-3kg':
        if idx == '10':
            t1 = t1 - 7.177 + 10.1
        elif idx == '11':
            t1 = t1 - 3.5745 + 8.866
        elif idx == '12':
            t1 = t1 - 5.055 + 9.283
        elif idx == '13':
            t1 = t1 - 4.588 + 6.983
        elif idx == '14':
            t1 = t1 - 9.098 + 12.816
    elif label == 'chenzui-left-all-4kg':
        if idx == '15':
            t1 = t1 - 3.3555 + 9.7
        elif idx == '16':
            t1 = t1 - 4.450 + 8.783
        elif idx == '17':
            t1 = t1 - 3.735 + 7.233
    elif label == 'chenzui-left-all-6.5kg':
        if idx == '10':
            t1 = t1 - 4.869 + 6.183
        elif idx == '11':
            t1 = t1 - 4.0085 + 6.033
        elif idx == '12':
            t1 = t1 - 6.505 + 8.0
        elif idx == '13':
            t1 = t1 - 3.8335 + 6.65
        elif idx == '14':
            t1 = t1 - 13.803 + 15.4

    # plt.figure(figsize=(6, 7.7))
    # plt.subplot(311)
    # plt.plot(t1, np.asarray(emg_BIC), label='emg', linewidth=2, zorder=3)
    # plt.ylabel('bic_s_l', weight='bold')
    # plt.legend()
    # plt.subplot(312)
    # plt.plot(t1, np.asarray(emg_BRA), label='emg', linewidth=2, zorder=3)
    # plt.ylabel('brachialis_1_l', weight='bold')
    # plt.legend()
    # plt.subplot(313)
    # plt.plot(t1, np.asarray(emg_BRD), label='emg', linewidth=2, zorder=3)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('brachiorad_1_l', weight='bold')
    # plt.legend()

    emg = [([]) for _ in range(len(muscle_idx))]
    time_emg = [[]]
    for t in t_tor_out[0]:
        idx = find_nearest_idx(t1, t)
        time_emg[0].append(t1[idx])
        emg[0].append(emg_BIC[idx])
        emg[1].append(emg_BRA[idx])
        emg[2].append(emg_BRD[idx])
        if include_TRI is True:
            emg[3].append(emg_TRI[idx])
        # emg[3].append(emg_TRIlong[idx])
        # emg[4][i].append(emg_TRIlat[idx])

    # plt.figure(figsize=(6, 7.7))
    # plt.subplot(311)
    # plt.plot(time_emg[0], np.asarray(emg[0]), label='emg', linewidth=2, zorder=3)
    # plt.ylabel('bic_s_l', weight='bold')
    # plt.legend()
    #
    # plt.subplot(312)
    # plt.plot(time_emg[0], np.asarray(emg[1]), label='emg', linewidth=2, zorder=3)
    # # plt.xlabel('time (s)')
    # plt.ylabel('brachialis_1_l', weight='bold')
    # plt.legend()
    #
    # plt.subplot(313)
    # plt.plot(time_emg[0], np.asarray(emg[2]), label='emg', linewidth=2, zorder=3)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('brachiorad_1_l', weight='bold')
    # plt.legend()
    # plt.show()

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
    # # if torque_init_0 is True:
    # if label == '11' or label == '13' or label == '14' or label == '15' or label == '17' or label == '18':
    #     tor_out = tor_out[0] - tor_out[0][0]
    emg_trend_u_out = np.asarray(emg_trend_u_out).squeeze()
    emg_trend_d_out = np.asarray(emg_trend_d_out).squeeze()
    emg_mean_out = np.asarray(emg_mean_out).squeeze()
    emg_std_out = np.asarray(emg_std_out).squeeze()
    arm_out = np.asarray(arm_out).squeeze()
    # fa_out = np.asarray(fa_out).squeeze()
    tor_out = np.asarray(tor_out).squeeze()
    t_tor_out = np.asarray(t_tor_out).squeeze()
    emg_mean = np.asarray(emg_mean).squeeze()
    emg = np.asarray(emg).squeeze()
    time_emg = np.asarray(time_emg).squeeze()
    # return emg_mean_out, emg_std_out, arm_out, fa_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    return emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out


def emg_file_progressing(emg, fs=1000, people=None):
    [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people)
    [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', people)
    [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', people)
    if include_TRI is True:
        [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', people)
        emg_list = np.asarray([emg_BIC, emg_BRA, emg_BRD, emg_TRI])
        t_list = np.asarray([t1, t2, t3, t4])
    else:
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
    yt = [([], [], [], [], []) for _ in range(3)]  # number of muscle
    for j in range(len(muscle_idx)):
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

    plt.figure(figsize=(6, 6.7))
    data = [([]) for _ in range(len(muscle_idx))]
    for k in range(len(muscle_idx)):
        for i in range(3):  # three weights
            for j in range(3):  # three actions
                data[k].append(np.concatenate([yt[i, k, int(2 * j), :], yt[i, k, int(2 * j + 1), :]]))
    data = np.asarray(data)
    data_mean = np.ones([data.shape[0], data.shape[2]])
    data_std = np.ones([data.shape[0], data.shape[2]])
    for i in range(len(muscle_idx)):  # three muscles
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
    color = ['#1f77b4', '#ff7f0e', '#2ca02c']
    label = ['10kg', '12.5kg', '15kg']
    plt.subplot(511)
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % 3])
    plt.ylabel('bic', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(512)
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % 3])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(513)
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % 3])
    plt.ylabel('brachiorad', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(514)
    for i in range(data.shape[1]):
        plt.plot(data[3, i, :], color=color[i % 3])
    plt.ylabel('tric_long', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(515)
    for i in range(data.shape[1]):
        if i < 3:
            plt.plot(data[4, i, :], color=color[i % 3], label=label[i])
        else:
            plt.plot(data[4, i, :], color=color[i % 3])
    plt.legend()
    plt.ylabel('tric_lat', weight='bold')
    plt.xlabel('timestep', weight='bold')

    plt.figure(figsize=(6, 6.7))
    plt.subplot(511)
    plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
    plt.ylabel('bic', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(512)
    plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(513)
    plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2])
    plt.ylabel('brachiorad', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(514)
    plt.errorbar(range(data_mean.shape[1]), data_mean[3], 2 * data_std[3])
    plt.ylabel('tric_long', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(515)
    plt.errorbar(range(data_mean.shape[1]), data_mean[4], 2 * data_std[4])
    plt.ylabel('tric_lat', weight='bold')
    plt.xlabel('timestep', weight='bold')

    plt.figure(figsize=(6, 6.7))
    plt.subplot(511)
    plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % 3])
    plt.ylabel('bic', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(512)
    plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % 3])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(513)
    plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % 3])
    plt.ylabel('brachiorad', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(514)
    plt.errorbar(range(data_mean.shape[1]), data_mean[3], 2 * data_std[3], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[3, i, :], color=color[i % 3])
    plt.ylabel('tric_long', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(515)
    plt.errorbar(range(data_mean.shape[1]), data_mean[4], 2 * data_std[4], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[4, i, :], color=color[i % 3])
    plt.ylabel('tric_lat', weight='bold')
    plt.xlabel('timestep', weight='bold')

    np.save('emg/yuetian_mean', data_mean)
    np.save('emg/yuetian_std', data_std)

    # print(np.max(yt[:, 0, :, :]))
    # print(np.max(yt[:, 1, :, :]))
    # print(np.max(yt[:, 2, :, :]))
    # print(np.max(yt[:, 3, :, :]))
    # print(np.max(yt[:, 4, :, :]))
    plt.show()


# def calculate_chenzui_emg_distribution(label='3kg'):
#     unified_len = 1000
#     fs = 2000
#     data_set_number = 7
#     if label == '3kg':
#         cz_11 = np.load('files/chenzui-3kg/emg-11.npy')
#         cz_13 = np.load('files/chenzui-3kg/emg-13.npy')
#         cz_14 = np.load('files/chenzui-3kg/emg-14.npy')
#         cz_15 = np.load('files/chenzui-3kg/emg-15.npy')
#         cz_16 = np.load('files/chenzui-3kg/emg-16.npy')
#         cz_17 = np.load('files/chenzui-3kg/emg-17.npy')
#         cz_18 = np.load('files/chenzui-3kg/emg-18.npy')
#     elif label == '5.5kg':
#         # cz_11 = np.load('files/chenzui-5.5kg/2.npy')
#         cz_11 = np.load('files/chenzui-5.5kg/7.npy')
#         cz_13 = np.load('files/chenzui-5.5kg/3.npy')
#         cz_14 = np.load('files/chenzui-5.5kg/4.npy')
#         cz_15 = np.load('files/chenzui-5.5kg/5.npy')
#         cz_16 = np.load('files/chenzui-5.5kg/6.npy')
#         cz_17 = np.load('files/chenzui-5.5kg/8.npy')
#         cz_18 = np.load('files/chenzui-5.5kg/9.npy')
#
#     [cz_11, t_cz_11] = emg_file_progressing(cz_11, fs, 'chenzui')
#     [cz_13, t_cz_13] = emg_file_progressing(cz_13, fs, 'chenzui')
#     [cz_14, t_cz_14] = emg_file_progressing(cz_14, fs, 'chenzui')
#     [cz_15, t_cz_15] = emg_file_progressing(cz_15, fs, 'chenzui')
#     [cz_16, t_cz_16] = emg_file_progressing(cz_16, fs, 'chenzui')
#     [cz_17, t_cz_17] = emg_file_progressing(cz_17, fs, 'chenzui')
#     [cz_18, t_cz_18] = emg_file_progressing(cz_18, fs, 'chenzui')
#     if label == '3kg':
#         t_cz_11 = t_cz_11 - 4.131 + 45.182  # 11
#         t_cz_13 = t_cz_13 - 4.763 + 209.74  # 13
#         t_cz_14 = t_cz_14 - 7.0005 + 289.653  # 14
#         t_cz_15 = t_cz_15 - 14.9555 + 391.032  # 15
#         t_cz_16 = t_cz_16 - 4.7075 + 482.544  # 16
#         t_cz_17 = t_cz_17 - 4.426 + 563.707  # 17
#         t_cz_18 = t_cz_18 - 4.03 + 638.553  # 18
#         timestep_emg_11 = [50.398, 57.531, 57.931, 70.13]
#         timestep_emg_13 = [214.69, 223.09, 223.09, 236.139]
#         timestep_emg_14 = [295.103, 303.336, 303.686, 316.469]
#         timestep_emg_15 = [395.348, 402.648, 403.398, 415.981]
#         timestep_emg_16 = [487.627, 495.31, 495.493, 509.676]
#         timestep_emg_17 = [568.907, 577.106, 577.106, 590.106]
#         timestep_emg_18 = [644.553, 653.669, 653.669, 668.352]
#     elif label == '5.5kg':
#         # t_cz_11 = t_cz_11 - 3.638 + 12.099
#         t_cz_11 = t_cz_11 - 3.169 + 13.967
#         t_cz_13 = t_cz_13 - 3.0785 + 11.783
#         t_cz_14 = t_cz_14 - 4.427 + 12.933
#         t_cz_15 = t_cz_15 - 5.5625 + 14.066
#         t_cz_16 = t_cz_16 - 5.274 + 11.65
#         # t_cz_17 = t_cz_17 - 3.169 + 13.967
#         t_cz_17 = t_cz_17 - 5.7455 + 12.833
#         t_cz_18 = t_cz_18 - 7.0975 + 15.699
#         # timestep_emg_11 = [17.582, 28.415, 28.415, 43.481]
#         timestep_emg_11 = [20.15, 30.299, 30.932, 44.298]
#         timestep_emg_13 = [17.033, 27.482, 27.482, 42.531]
#         timestep_emg_14 = [19.216, 29.799, 30.132, 44.398]
#         timestep_emg_15 = [19.382, 30.648, 30.648, 46.048]
#         timestep_emg_16 = [18.149, 28.249, 28.332, 42.515]
#         # timestep_emg_17 = [20.15, 30.299, 30.932, 44.298]
#         timestep_emg_17 = [19.432, 29.532, 29.782, 43.015]
#         timestep_emg_18 = [21.716, 32.449, 33.299, 49.748]
#
#     if include_TRI is True:
#         yt = [([], [], [], []) for _ in range(data_set_number)]  # number of muscle
#     else:
#         yt = [([], [], []) for _ in range(data_set_number)]  # number of muscle
#     for j in range(len(muscle_idx)):
#         for i in range(int(len(timestep_emg_11) / 2)):
#             yt[0][j].append(resample_by_len(cz_11[j, find_nearest_idx(t_cz_11, timestep_emg_11[2 * i]):
#                                                      find_nearest_idx(t_cz_11, timestep_emg_11[2 * i + 1])],
#                                             unified_len))
#             yt[1][j].append(resample_by_len(cz_13[j, find_nearest_idx(t_cz_13, timestep_emg_13[2 * i]):
#                                                      find_nearest_idx(t_cz_13, timestep_emg_13[2 * i + 1])],
#                                             unified_len))
#             yt[2][j].append(resample_by_len(cz_14[j, find_nearest_idx(t_cz_14, timestep_emg_14[2 * i]):
#                                                      find_nearest_idx(t_cz_14, timestep_emg_14[2 * i + 1])],
#                                             unified_len))
#             yt[3][j].append(resample_by_len(cz_15[j, find_nearest_idx(t_cz_15, timestep_emg_15[2 * i]):
#                                                      find_nearest_idx(t_cz_15, timestep_emg_15[2 * i + 1])],
#                                             unified_len))
#             yt[4][j].append(resample_by_len(cz_16[j, find_nearest_idx(t_cz_16, timestep_emg_16[2 * i]):
#                                                      find_nearest_idx(t_cz_16, timestep_emg_16[2 * i + 1])],
#                                             unified_len))
#             yt[5][j].append(resample_by_len(cz_17[j, find_nearest_idx(t_cz_17, timestep_emg_17[2 * i]):
#                                                      find_nearest_idx(t_cz_17, timestep_emg_17[2 * i + 1])],
#                                             unified_len))
#             yt[6][j].append(resample_by_len(cz_18[j, find_nearest_idx(t_cz_18, timestep_emg_18[2 * i]):
#                                                      find_nearest_idx(t_cz_18, timestep_emg_18[2 * i + 1])],
#                                             unified_len))
#
#     yt = np.asarray(yt)
#     data = [([]) for _ in range(len(muscle_idx))]
#     for k in range(len(muscle_idx)):
#         for i in range(data_set_number):
#             data[k].append(np.concatenate([yt[i, k, 0, :], yt[i, k, 1, :]]))
#     data = np.asarray(data)
#
#     NUMPASSES = 3
#     LOWPASSRATE = 3  # 低通滤波4—10Hz得到包络线
#     Fs = 200
#     Wn = LOWPASSRATE / (Fs / 2)
#     [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
#     data = scipy.signal.filtfilt(b, a, data, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波
#
#     data_mean = np.ones([yt.shape[0], data.shape[2]])
#     data_std = np.ones([data.shape[0], data.shape[2]])
#     data_trend_u = np.ones([data.shape[0], data.shape[2] - 1])
#     data_trend_d = np.ones([data.shape[0], data.shape[2] - 1])
#     for i in range(len(muscle_idx)):  # three muscles
#         data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
#         data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
#         # data_trend_u[i] = np.asarray([np.max(data[i, :, j + 1] / data[i, :, j]) for j in range(data.shape[2] - 1)])
#         # data_trend_d[i] = np.asarray([np.min(data[i, :, j + 1] / data[i, :, j]) for j in range(data.shape[2] - 1)])
#         data_trend_u[i] = np.asarray(
#             [(np.max(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
#         data_trend_d[i] = np.asarray(
#             [(np.min(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
#         # data_trend_u[i] = np.asarray([np.max(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])
#         # data_trend_d[i] = np.asarray([np.min(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])
#
#     # data_mean = np.ones([data.shape[0], data.shape[2]])
#     # data_std = np.ones([data.shape[0], data.shape[2]])
#     # for i in range(len(muscle_idx)):  # three muscles
#     #     data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
#     #     data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
#
#     plt.figure(figsize=(6, 6.7))
#     color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#     plt.subplot(311)
#     for i in range(data.shape[1]):
#         plt.plot(data[0, i, :], color=color[i % len(color)])
#     plt.ylabel('bic', weight='bold')
#     ax = plt.gca()
#     ax.axes.xaxis.set_visible(False)
#     plt.subplot(312)
#     for i in range(data.shape[1]):
#         plt.plot(data[1, i, :], color=color[i % len(color)])
#     plt.ylabel('brachialis', weight='bold')
#     ax = plt.gca()
#     ax.axes.xaxis.set_visible(False)
#     plt.subplot(313)
#     for i in range(data.shape[1]):
#         plt.plot(data[2, i, :], color=color[i % len(color)])
#     # for i in range(data.shape[1]):
#     #     if i < 3:
#     #         plt.plot(data[2, i, :], color=color[i % 3], label=label[i])
#     #     else:
#     #         plt.plot(data[2, i, :], color=color[i % 3])
#     # plt.legend()
#     plt.ylabel('brachiorad', weight='bold')
#     plt.xlabel('timestep', weight='bold')
#
#     if include_TRI is False:
#         plt.figure(figsize=(6, 6.7))
#         plt.subplot(311)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
#         plt.ylabel('bic', weight='bold')
#         ax = plt.gca()
#         ax.axes.xaxis.set_visible(False)
#         plt.subplot(312)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1])
#         plt.ylabel('brachialis', weight='bold')
#         ax = plt.gca()
#         ax.axes.xaxis.set_visible(False)
#         plt.subplot(313)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2])
#         plt.ylabel('brachiorad', weight='bold')
#         plt.xlabel('timestep', weight='bold')
#     else:
#         plt.figure(figsize=(6, 6.7))
#         plt.subplot(411)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
#         plt.ylabel('bic', weight='bold')
#         ax = plt.gca()
#         ax.axes.xaxis.set_visible(False)
#         plt.subplot(412)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1])
#         plt.ylabel('brachialis', weight='bold')
#         ax = plt.gca()
#         ax.axes.xaxis.set_visible(False)
#         plt.subplot(413)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2])
#         plt.ylabel('brachiorad', weight='bold')
#         ax = plt.gca()
#         ax.axes.xaxis.set_visible(False)
#         plt.subplot(414)
#         plt.errorbar(range(data_mean.shape[1]), data_mean[3], 2 * data_std[3])
#         plt.ylabel('tri', weight='bold')
#         plt.xlabel('timestep', weight='bold')
#
#     plt.figure(figsize=(6, 6.7))
#     plt.subplot(311)
#     plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0], color='papayawhip')
#     for i in range(data.shape[1]):
#         plt.plot(data[0, i, :], color=color[i % len(color)])
#     plt.ylabel('bic', weight='bold')
#     ax = plt.gca()
#     ax.axes.xaxis.set_visible(False)
#     plt.subplot(312)
#     plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1], color='papayawhip')
#     for i in range(data.shape[1]):
#         plt.plot(data[1, i, :], color=color[i % len(color)])
#     plt.ylabel('brachialis', weight='bold')
#     ax = plt.gca()
#     ax.axes.xaxis.set_visible(False)
#     plt.subplot(313)
#     plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2], color='papayawhip')
#     for i in range(data.shape[1]):
#         plt.plot(data[2, i, :], color=color[i % len(color)])
#     plt.xlabel('timestep', weight='bold')
#
#     if label == '3kg':
#         np.save('emg/chenzui_mean_3kg', data_mean)
#         np.save('emg/chenzui_std_3kg', data_std)
#         np.save('emg/chenzui_trend_u_3kg', data_trend_u)
#         np.save('emg/chenzui_trend_d_3kg', data_trend_d)
#     elif label == '5.5kg':
#         np.save('emg/chenzui_mean_5.5kg', data_mean)
#         np.save('emg/chenzui_std_5.5kg', data_std)
#         np.save('emg/chenzui_trend_u_3kg', data_trend_u)
#         np.save('emg/chenzui_trend_d_3kg', data_trend_d)
#
#     # print(np.max(yt[:, 0, :, :]))
#     # print(np.max(yt[:, 1, :, :]))
#     # print(np.max(yt[:, 2, :, :]))
#     # print(np.max(yt[:, 3, :, :]))
#     # print(np.max(yt[:, 4, :, :]))
#     plt.show()

def calculate_chenzui_emg_distribution(label='1kg'):
    unified_len = 1000
    fs = 2000
    if label == '3kg':
        files = ['files/chenzui-3kg/emg-11.npy',
                 'files/chenzui-3kg/emg-13.npy',
                 'files/chenzui-3kg/emg-14.npy',
                 'files/chenzui-3kg/emg-15.npy',
                 'files/chenzui-3kg/emg-16.npy',
                 'files/chenzui-3kg/emg-17.npy',
                 'files/chenzui-3kg/emg-18.npy']
        t_delta_emg = [4.131, 4.763, 7.0005, 14.9555, 4.7075, 4.426, 4.03]
        t_delta_joi = [45.182, 209.74, 289.653, 391.032, 482.544, 563.707, 638.553]
        timestep_emg = [[50.398, 57.531, 57.931, 70.13],
                        [214.69, 223.09, 223.09, 236.139],
                        [295.103, 303.336, 303.686, 316.469],
                        [395.348, 402.648, 403.398, 415.981],
                        [487.627, 495.31, 495.493, 509.676],
                        [568.907, 577.106, 577.106, 590.106],
                        [644.553, 653.669, 653.669, 668.352]]
    elif label == '5.5kg':
        files = ['files/chenzui-5.5kg/7.npy',
                 'files/chenzui-5.5kg/3.npy',
                 'files/chenzui-5.5kg/4.npy',
                 'files/chenzui-5.5kg/5.npy',
                 'files/chenzui-5.5kg/6.npy',
                 'files/chenzui-5.5kg/8.npy',
                 'files/chenzui-5.5kg/9.npy',]
        t_delta_emg = [3.169, 3.0785, 4.427, 5.5625, 5.274, 5.7455, 7.0975]
        t_delta_joi = [13.967, 11.783, 12.933, 14.066, 11.65, 12.833, 15.699]
        timestep_emg = [[20.15, 30.299, 30.932, 44.298],
                        [17.033, 27.482, 27.482, 42.531],
                        [19.216, 29.799, 30.132, 44.398],
                        [19.382, 30.648, 30.648, 46.048],
                        [18.149, 28.249, 28.332, 42.515],
                        [19.432, 29.532, 29.782, 43.015],
                        [21.716, 32.449, 33.299, 49.748]]
    elif label == '2.5kg-all':
        files = ['files/chenzui-all/2.5-5.npy',
                 'files/chenzui-all/2.5-6.npy',
                 'files/chenzui-all/2.5-7.npy',
                 'files/chenzui-all/2.5-8.npy',
                 'files/chenzui-all/2.5-9.npy']
        t_delta_emg = [3.464, 3.7925, 3.2645, 3.178, 4.6665]
        t_delta_joi = [9.733, 7.65, 8.583, 11.249, 9.466]
        timestep_emg = [[16.416, 27.065, 27.065, 41.681],
                        [13.750, 24.782, 24.782, 42.731],
                        [13.649, 25.982, 25.982, 41.598],
                        [15.832, 28.482, 28.482, 44.864],
                        [15.549, 26.365, 26.599, 40.031]]
    elif label == '3kg-all':
        files = ['files/chenzui-all/3-10.npy',
                 'files/chenzui-all/3-11.npy',
                 'files/chenzui-all/3-12.npy',
                 'files/chenzui-all/3-13.npy',
                 'files/chenzui-all/3-14.npy']
        t_delta_emg = [7.177, 3.5745, 5.055, 4.588, 9.098]
        t_delta_joi = [10.1, 8.866, 9.283, 6.983, 12.816]
        timestep_emg = [[15.599, 26.249, 26.532, 39.348],
                        [15.216, 25.582, 25.582, 40.781],
                        [16.066, 28.299, 28.299, 40.882],
                        [12.049, 22.699, 22.699, 34.965],
                        [17.849, 29.365, 29.365, 43.381]]
    elif label == '4kg-all':
        files = ['files/chenzui-all/4-15.npy',
                 'files/chenzui-all/4-16.npy',
                 'files/chenzui-all/4-17.npy',]
        t_delta_emg = [3.3555, 4.450, 3.735]
        t_delta_joi = [9.7, 8.783, 7.233]
        timestep_emg = [[14.966, 24.216, 24.216, 37.399],
                        [13.133, 23.899, 23.899, 37.565],
                        [11.433, 22.516, 22.516, 35.682]]
    elif label == '6.5kg-all':
        files = ['files/chenzui-all/6.5-18.npy',
                 'files/chenzui-all/6.5-19.npy',
                 'files/chenzui-all/6.5-20.npy',
                 'files/chenzui-all/6.5-21.npy',
                 'files/chenzui-all/6.5-22.npy']
        t_delta_emg = [4.869, 4.0085, 6.505, 3.8335, 13.803]
        t_delta_joi = [6.183, 6.033, 8.0, 6.65, 15.4]
        timestep_emg = [[11.233, 23.465, 23.465, 37.948],
                        [11.833, 25.132, 25.132, 36.548],
                        [14.633, 27.399, 27.399, 40.465],
                        [11.9, 24.599, 24.599, 37.465],
                        [19.466, 30.415, 31.749, 45.781]]  # 6.5kg
    elif label == 'all-1':
        files = ['files/chenzui-all/2.5-5.npy',
                 'files/chenzui-all/2.5-6.npy',
                 'files/chenzui-all/2.5-7.npy',
                 'files/chenzui-all/2.5-8.npy',
                 'files/chenzui-all/2.5-9.npy',
                 'files/chenzui-all/3-10.npy',
                 'files/chenzui-all/3-11.npy',
                 'files/chenzui-all/3-12.npy',
                 'files/chenzui-all/3-13.npy',
                 'files/chenzui-all/3-14.npy',
                 'files/chenzui-all/4-15.npy',
                 'files/chenzui-all/4-16.npy',
                 'files/chenzui-all/4-17.npy',
                 'files/chenzui-all/6.5-18.npy',
                 'files/chenzui-all/6.5-19.npy',
                 'files/chenzui-all/6.5-20.npy',
                 'files/chenzui-all/6.5-21.npy',
                 'files/chenzui-all/6.5-22.npy']
        t_delta_emg = [3.464, 3.7925, 3.2645, 3.178, 4.6665,
                       7.177, 3.5745, 5.055, 4.588, 9.098,
                       3.3555, 4.450, 3.735,
                       4.869, 4.0085, 6.505, 3.8335, 13.803]
        t_delta_joi = [9.733, 7.65, 8.583, 11.249, 9.466,
                       10.1, 8.866, 9.283, 6.983, 12.816,
                       9.7, 8.783, 7.233,
                       6.183, 6.033, 8.0, 6.65, 15.4]
        timestep_emg = [[16.416, 27.065, 27.065, 41.681],
                        [13.750, 24.782, 24.782, 42.731],
                        [13.649, 25.982, 25.982, 41.598],
                        [15.832, 28.482, 28.482, 44.864],
                        [15.549, 26.365, 26.599, 40.031], # 2.5kg
                        [15.599, 26.249, 26.532, 39.348],
                        [15.216, 25.582, 25.582, 40.781],
                        [16.066, 28.299, 28.299, 40.882],
                        [12.049, 22.699, 22.699, 34.965],
                        [17.849, 29.365, 29.365, 43.381], # 3kg
                        [14.966, 24.216, 24.216, 37.399],
                        [13.133, 23.899, 23.899, 37.565],
                        [11.433, 22.516, 22.516, 35.682], # 4kg
                        [11.233, 23.465, 23.465, 37.948],
                        [11.833, 25.132, 25.132, 36.548],
                        [14.633, 27.399, 27.399, 40.465],
                        [11.9, 24.599, 24.599, 37.465],
                        [19.466, 30.415, 31.749, 45.781]] # 6.5kg
    elif label == 'all':
        files = ['files/chenzui-all/2.5-5.npy',
                 'files/chenzui-all/2.5-6.npy',
                 'files/chenzui-all/2.5-7.npy',
                 'files/chenzui-all/2.5-8.npy',
                 'files/chenzui-all/2.5-9.npy',
                 'files/chenzui-all/3-10.npy',
                 'files/chenzui-all/3-11.npy',
                 'files/chenzui-all/3-12.npy',
                 'files/chenzui-all/3-13.npy',
                 'files/chenzui-all/3-14.npy',
                 'files/chenzui-all/4-15.npy',
                 'files/chenzui-all/4-16.npy',
                 'files/chenzui-all/4-17.npy']
        t_delta_emg = [3.464, 3.7925, 3.2645, 3.178, 4.6665,
                       7.177, 3.5745, 5.055, 4.588, 9.098,
                       3.3555, 4.450, 3.735]
        t_delta_joi = [9.733, 7.65, 8.583, 11.249, 9.466,
                       10.1, 8.866, 9.283, 6.983, 12.816,
                       9.7, 8.783, 7.233]
        timestep_emg = [[16.416, 27.065, 27.065, 41.681],
                        [13.750, 24.782, 24.782, 42.731],
                        [13.649, 25.982, 25.982, 41.598],
                        [15.832, 28.482, 28.482, 44.864],
                        [15.549, 26.365, 26.599, 40.031], # 2.5kg
                        [15.599, 26.249, 26.532, 39.348],
                        [15.216, 25.582, 25.582, 40.781],
                        [16.066, 28.299, 28.299, 40.882],
                        [12.049, 22.699, 22.699, 34.965],
                        [17.849, 29.365, 29.365, 43.381], # 3kg
                        [14.966, 24.216, 24.216, 37.399],
                        [13.133, 23.899, 23.899, 37.565],
                        [11.433, 22.516, 22.516, 35.682]] # 4kg
    else:
        print('No label:', label)

    data_set_number = len(files)
    # if label == '3kg' or label == '5.5kg':
    #     data_set_number = 7
    # elif label == '2.5kg':
    #     data_set_number = 5
    emg_all = [([]) for _ in range(data_set_number)]
    t_emg_all = [([]) for _ in range(data_set_number)]
    for i in range(data_set_number):
        emg_all[i] = np.load(files[i])
        [emg_all[i], t_emg_all[i]] = emg_file_progressing(emg_all[i], fs, 'zhuo')
        t_emg_all[i] = t_emg_all[i] - t_delta_emg[i] + t_delta_joi[i]

    if include_TRI is True:
        muscles = [([], [], [], []) for _ in range(data_set_number)]  # number of muscle
    else:
        muscles = [([], [], []) for _ in range(data_set_number)]  # number of muscle
    for k in range(data_set_number):
        for j in range(len(muscle_idx)):
            for i in range(int(len(timestep_emg[k]) / 2)):
                muscles[k][j].append(resample_by_len(emg_all[k][j, find_nearest_idx(t_emg_all[k], timestep_emg[k][
                    2 * i]): find_nearest_idx(t_emg_all[k], timestep_emg[k][2 * i + 1])], unified_len))

    muscles = np.asarray(muscles)

    data = [([]) for _ in range(len(muscle_idx))]
    for k in range(len(muscle_idx)):
        for i in range(data_set_number):
            data[k].append(np.concatenate([muscles[i, k, 0, :], muscles[i, k, 1, :]]))
    data = np.asarray(data)

    # NUMPASSES = 3
    # LOWPASSRATE = 3  # 低通滤波4—10Hz得到包络线
    # Fs = 400
    # Wn = LOWPASSRATE / (Fs / 2)
    # [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    # data = scipy.signal.filtfilt(b, a, data, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    # plt.figure(figsize=(6, 6.7))
    # plt.subplot(311)
    # plt.plot(data[0, 0, :])
    # plt.plot(data[0, 1, :])
    # plt.ylabel('biceps', weight='bold')
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # plt.subplot(312)
    # plt.plot(data[1, 0, :])
    # plt.plot(data[1, 1, :])
    # plt.ylabel('brachialis', weight='bold')
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # plt.subplot(313)
    # plt.plot(data[2, 0, :])
    # plt.plot(data[2, 1, :])
    # # plt.show()

    data_mean = np.ones([data.shape[0], data.shape[2]])
    data_std = np.ones([data.shape[0], data.shape[2]])
    data_trend_u = np.ones([data.shape[0], data.shape[2] - 1])
    data_trend_d = np.ones([data.shape[0], data.shape[2] - 1])
    for i in range(len(muscle_idx)):  # three muscles
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
        # data_trend_u[i] = np.asarray([np.max(data[i, :, j + 1] / data[i, :, j]) for j in range(data.shape[2] - 1)])
        # data_trend_d[i] = np.asarray([np.min(data[i, :, j + 1] / data[i, :, j]) for j in range(data.shape[2] - 1)])
        data_trend_u[i] = np.asarray([(np.max(data[i, :, j + 1] - data[i, :, j])/0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray([(np.min(data[i, :, j + 1] - data[i, :, j])/0.01) for j in range(data.shape[2] - 1)])
        # data_trend_u[i] = np.asarray([np.max(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])
        # data_trend_d[i] = np.asarray([np.min(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])

    # plt.figure(figsize=(6, 6.7))
    # plt.subplot(311)
    # # plt.plot(data_mean[0])
    # # plt.plot(np.hstack([data_mean[0, 0], data_mean[0, :-1] * data_trend_u[0]]))
    # # plt.plot(np.hstack([data_mean[0, 0], data_mean[0, :-1] * data_trend_d[0]]))
    # plt.plot(data_trend_u[0])
    # plt.plot(data_trend_d[0])
    # plt.ylabel('biceps', weight='bold')
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # plt.subplot(312)
    # # plt.plot(data_mean[1])
    # # plt.plot(np.hstack([data_mean[1, 0], data_mean[1, :-1] * data_trend_u[1]]))
    # # plt.plot(np.hstack([data_mean[1, 0], data_mean[1, :-1] * data_trend_d[1]]))
    # plt.plot(data_trend_u[1])
    # plt.plot(data_trend_d[1])
    # plt.ylabel('brachialis', weight='bold')
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # plt.subplot(313)
    # # plt.plot(data_mean[2])
    # # plt.plot(np.hstack([data_mean[2, 0], data_mean[2, :-1] * data_trend_u[2]]))
    # # plt.plot(np.hstack([data_mean[2, 0], data_mean[2, :-1] * data_trend_d[2]]))
    # plt.plot(data_trend_u[2])
    # plt.plot(data_trend_d[2])
    # plt.ylabel('brachiorad', weight='bold')
    # plt.xlabel('timestep', weight='bold')

    plt.figure(figsize=(6, 6.7))
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.subplot(311)
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % len(color)])
    plt.ylabel('biceps', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % len(color)])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % len(color)])
    # for i in range(data.shape[1]):
    #     if i < 3:
    #         plt.plot(data[2, i, :], color=color[i % 3], label=label[i])
    #     else:
    #         plt.plot(data[2, i, :], color=color[i % 3])
    # plt.legend()
    plt.ylabel('brachiorad', weight='bold')
    plt.xlabel('timestep', weight='bold')

    if include_TRI is False:
        plt.figure(figsize=(6, 6.7))
        plt.subplot(311)
        plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
        plt.ylabel('biceps', weight='bold')
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
    else:
        plt.figure(figsize=(6, 6.7))
        plt.subplot(411)
        plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
        plt.ylabel('bic', weight='bold')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.subplot(412)
        plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1])
        plt.ylabel('brachialis', weight='bold')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.subplot(413)
        plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2])
        plt.ylabel('brachiorad', weight='bold')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.subplot(414)
        plt.errorbar(range(data_mean.shape[1]), data_mean[3], 2 * data_std[3])
        plt.ylabel('tri', weight='bold')
        plt.xlabel('timestep', weight='bold')

    plt.figure(figsize=(6, 6.7))
    plt.subplot(311)
    plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % len(color)])
    plt.ylabel('biceps', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % len(color)])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % len(color)])
    plt.xlabel('timestep', weight='bold')

    if label == '2.5kg-all':
        np.save('emg/chenzui_mean_2.5kg', data_mean)
        np.save('emg/chenzui_std_2.5kg', data_std)
        np.save('emg/chenzui_trend_u_2.5kg', data_trend_u)
        np.save('emg/chenzui_trend_d_2.5kg', data_trend_d)
    elif label == '3kg-all':
        np.save('emg/chenzui_mean_3kg', data_mean)
        np.save('emg/chenzui_std_3kg', data_std)
        np.save('emg/chenzui_trend_u_3kg', data_trend_u)
        np.save('emg/chenzui_trend_d_3kg', data_trend_d)
    elif label == '4kg-all':
        np.save('emg/chenzui_mean_4kg', data_mean)
        np.save('emg/chenzui_std_4kg', data_std)
        np.save('emg/chenzui_trend_u_4kg', data_trend_u)
        np.save('emg/chenzui_trend_d_4kg', data_trend_d)
    elif label == '6.5kg-all':
        np.save('emg/chenzui_mean_6.5kg', data_mean)
        np.save('emg/chenzui_std_6.5kg', data_std)
        np.save('emg/chenzui_trend_u_6.5kg', data_trend_u)
        np.save('emg/chenzui_trend_d_6.5kg', data_trend_d)
    elif label == '3kg':
        np.save('emg/chenzui_mean_3kg', data_mean)
        np.save('emg/chenzui_std_3kg', data_std)
        np.save('emg/chenzui_trend_u_3kg', data_trend_u)
        np.save('emg/chenzui_trend_d_3kg', data_trend_d)
    elif label == '5.5kg':
        np.save('emg/chenzui_mean_5.5kg', data_mean)
        np.save('emg/chenzui_std_5.5kg', data_std)
        np.save('emg/chenzui_trend_u_5.5kg', data_trend_u)
        np.save('emg/chenzui_trend_d_5.5kg', data_trend_d)
    elif label == 'all' or label == 'all-1':
        np.save('emg/chenzui_mean_all', data_mean)
        np.save('emg/chenzui_std_all', data_std)
        np.save('emg/chenzui_trend_u_all', data_trend_u)
        np.save('emg/chenzui_trend_d_all', data_trend_d)

    # print(np.max(yt[:, 0, :, :]))
    # print(np.max(yt[:, 1, :, :]))
    # print(np.max(yt[:, 2, :, :]))
    # print(np.max(yt[:, 3, :, :]))
    # print(np.max(yt[:, 4, :, :]))
    plt.show()


def calculate_lizhuo_emg_distribution(label='1kg'):
    unified_len = 1000
    fs = 2000
    data_set_number = 7
    emg_all = [([]) for _ in range(data_set_number)]
    t_emg_all = [([]) for _ in range(data_set_number)]
    if label == '1kg':
        files = ['files/lizhuo-1kg/2.npy',
                 'files/lizhuo-1kg/3.npy',
                 'files/lizhuo-1kg/4.npy',
                 'files/lizhuo-1kg/6.npy',
                 'files/lizhuo-1kg/7.npy',
                 'files/lizhuo-1kg/8.npy',
                 'files/lizhuo-1kg/9.npy']
        t_delta_emg = [11.3095, 4.0, 4.3385, 4.9365, 4.8945, 3.5095, 4.2645]
        t_delta_joi = [16.0, 12.149, 16.416, 11.283, 11.033, 8.433, 9.899]
        timestep_emg = [[20.016, 27.599, 27.599, 37.465],
                        [14.365, 25.798, 25.798, 36.248],
                        [12.766, 27.032, 27.032, 39.865],
                        [13.483, 25.832, 25.866, 55.331],
                        [12.566, 29.299, 29.299, 53.014],
                        [9.55, 33.932, 33.932, 56.414],
                        [12.199, 29.698, 29.698, 51.514]]
    elif label == '3kg':
        files = ['files/lizhuo-3kg/1.npy',
                 'files/lizhuo-3kg/2.npy',
                 'files/lizhuo-3kg/3.npy',
                 'files/lizhuo-3kg/5.npy',
                 'files/lizhuo-3kg/6.npy',
                 # 'files/lizhuo-3kg/7.npy',
                 'files/lizhuo-3kg/8.npy',
                 'files/lizhuo-3kg/9.npy',
                 'files/lizhuo-3kg/10.npy',
                 'files/lizhuo-3kg/11.npy',
                 'files/lizhuo-3kg/12.npy']
        # t_delta_emg = [4.2525, 8.14, 3.6615, 3.642, 4.103, 7.3255, 3.873, 4.2335, 4.1045, 5.6695, 7.29]
        # t_delta_joi = [13.067, 14.1, 9.866, 7.749, 9.716, 13.183, 9.866, 9.75, 9.367, 12.766, 14.749]
        t_delta_emg = [4.2525, 8.14, 3.6615, 3.642, 4.103, 3.873, 4.2335, 4.1045, 5.6695, 7.29]
        t_delta_joi = [13.067, 14.1, 9.866, 7.749, 9.716, 9.866, 9.75, 9.367, 12.766, 14.749]
        timestep_emg = [[20.383, 32.832, 32.832, 46.348],
                        [19.250, 32.449, 32.449, 47.765],
                        [14.866, 29.916, 29.916, 48.015],
                        [12.099, 24.215, 25.932, 42.098],
                        [14.133, 30.015, 30.015, 50.098],
                        # [15.933, 24.233, 24.233, 37.882],
                        [15.332, 25.449, 25.449, 39.265],
                        [13.7, 27.116, 27.116, 37.649],
                        [12.8, 20.0, 20.0, 28.799],  # 7.20s, 8.80s
                        [16.099, 25.332, 25.332, 37.848],
                        [19.032, 21.649, 21.649, 24.465],  # 2.62s, 2.82s
                        [24.465, 26.865, 26.865, 30.165],  # 2.40s, 3.30s
                        [30.665, 33.198, 33.198, 36.498]]
    elif label == '3kg-f':
        files = ['files/lizhuo-3kg/12.npy',
                 'files/lizhuo-3kg/12.npy',
                 'files/lizhuo-3kg/12.npy']
        t_delta_emg = [7.29, 7.29, 7.29]
        t_delta_joi = [14.749, 14.749, 14.749]
        timestep_emg = [[19.032, 21.649, 21.649, 24.465],  # 2.62s, 2.82s
                        [24.465, 26.865, 26.865, 30.165],  # 2.40s, 3.30s
                        [30.665, 33.198, 33.198, 36.498]]  # 2.53s, 3.30s
    else:
        print('No label:', label)
    for i in range(data_set_number):
        emg_all[i] = np.load(files[i])
        [emg_all[i], t_emg_all[i]] = emg_file_progressing(emg_all[i], fs, 'zhuo')
        t_emg_all[i] = t_emg_all[i] - t_delta_emg[i] + t_delta_joi[i]

    if include_TRI is True:
        muscles = [([], [], [], []) for _ in range(data_set_number)]  # number of muscle
    else:
        muscles = [([], [], []) for _ in range(data_set_number)]  # number of muscle
    for k in range(data_set_number):
        for j in range(len(muscle_idx)):
            for i in range(int(len(timestep_emg[k]) / 2)):
                muscles[k][j].append(resample_by_len(emg_all[k][j, find_nearest_idx(t_emg_all[k], timestep_emg[k][
                    2 * i]): find_nearest_idx(t_emg_all[k], timestep_emg[k][2 * i + 1])], unified_len))

    muscles = np.asarray(muscles)

    data = [([]) for _ in range(len(muscle_idx))]
    for k in range(len(muscle_idx)):
        for i in range(data_set_number):
            data[k].append(np.concatenate([muscles[i, k, 0, :], muscles[i, k, 1, :]]))
    data = np.asarray(data)

    NUMPASSES = 3
    LOWPASSRATE = 3  # 低通滤波4—10Hz得到包络线
    Fs = 200
    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    data = scipy.signal.filtfilt(b, a, data, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    plt.figure(figsize=(6, 6.7))
    plt.subplot(311)
    plt.plot(data[0, 0, :])
    plt.plot(data[0, 1, :])
    plt.ylabel('biceps', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    plt.plot(data[1, 0, :])
    plt.plot(data[1, 1, :])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    plt.plot(data[2, 0, :])
    plt.plot(data[2, 1, :])
    # plt.show()

    data_mean = np.ones([data.shape[0], data.shape[2]])
    data_std = np.ones([data.shape[0], data.shape[2]])
    data_trend_u = np.ones([data.shape[0], data.shape[2] - 1])
    data_trend_d = np.ones([data.shape[0], data.shape[2] - 1])
    for i in range(len(muscle_idx)):  # three muscles
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
        # data_trend_u[i] = np.asarray([np.max(data[i, :, j + 1] / data[i, :, j]) for j in range(data.shape[2] - 1)])
        # data_trend_d[i] = np.asarray([np.min(data[i, :, j + 1] / data[i, :, j]) for j in range(data.shape[2] - 1)])
        data_trend_u[i] = np.asarray([(np.max(data[i, :, j + 1] - data[i, :, j])/0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray([(np.min(data[i, :, j + 1] - data[i, :, j])/0.01) for j in range(data.shape[2] - 1)])
        # data_trend_u[i] = np.asarray([np.max(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])
        # data_trend_d[i] = np.asarray([np.min(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])

    plt.figure(figsize=(6, 6.7))
    plt.subplot(311)
    # plt.plot(data_mean[0])
    # plt.plot(np.hstack([data_mean[0, 0], data_mean[0, :-1] * data_trend_u[0]]))
    # plt.plot(np.hstack([data_mean[0, 0], data_mean[0, :-1] * data_trend_d[0]]))
    plt.plot(data_trend_u[0])
    plt.plot(data_trend_d[0])
    plt.ylabel('biceps', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    # plt.plot(data_mean[1])
    # plt.plot(np.hstack([data_mean[1, 0], data_mean[1, :-1] * data_trend_u[1]]))
    # plt.plot(np.hstack([data_mean[1, 0], data_mean[1, :-1] * data_trend_d[1]]))
    plt.plot(data_trend_u[1])
    plt.plot(data_trend_d[1])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    # plt.plot(data_mean[2])
    # plt.plot(np.hstack([data_mean[2, 0], data_mean[2, :-1] * data_trend_u[2]]))
    # plt.plot(np.hstack([data_mean[2, 0], data_mean[2, :-1] * data_trend_d[2]]))
    plt.plot(data_trend_u[2])
    plt.plot(data_trend_d[2])
    plt.ylabel('brachiorad', weight='bold')
    plt.xlabel('timestep', weight='bold')

    plt.figure(figsize=(6, 6.7))
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.subplot(311)
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % len(color)])
    plt.ylabel('biceps', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % len(color)])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % len(color)])
    # for i in range(data.shape[1]):
    #     if i < 3:
    #         plt.plot(data[2, i, :], color=color[i % 3], label=label[i])
    #     else:
    #         plt.plot(data[2, i, :], color=color[i % 3])
    # plt.legend()
    plt.ylabel('brachiorad', weight='bold')
    plt.xlabel('timestep', weight='bold')

    if include_TRI is False:
        plt.figure(figsize=(6, 6.7))
        plt.subplot(311)
        plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
        plt.ylabel('biceps', weight='bold')
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
    else:
        plt.figure(figsize=(6, 6.7))
        plt.subplot(411)
        plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0])
        plt.ylabel('bic', weight='bold')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.subplot(412)
        plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1])
        plt.ylabel('brachialis', weight='bold')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.subplot(413)
        plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2])
        plt.ylabel('brachiorad', weight='bold')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.subplot(414)
        plt.errorbar(range(data_mean.shape[1]), data_mean[3], 2 * data_std[3])
        plt.ylabel('tri', weight='bold')
        plt.xlabel('timestep', weight='bold')

    plt.figure(figsize=(6, 6.7))
    plt.subplot(311)
    plt.errorbar(range(data_mean.shape[1]), data_mean[0], 2 * data_std[0], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[0, i, :], color=color[i % len(color)])
    plt.ylabel('biceps', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(312)
    plt.errorbar(range(data_mean.shape[1]), data_mean[1], 2 * data_std[1], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[1, i, :], color=color[i % len(color)])
    plt.ylabel('brachialis', weight='bold')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.subplot(313)
    plt.errorbar(range(data_mean.shape[1]), data_mean[2], 2 * data_std[2], color='papayawhip')
    for i in range(data.shape[1]):
        plt.plot(data[2, i, :], color=color[i % len(color)])
    plt.xlabel('timestep', weight='bold')

    if label == '1kg':
        np.save('emg/lizhuo_mean_1kg', data_mean)
        np.save('emg/lizhuo_std_1kg', data_std)
        np.save('emg/lizhuo_trend_u_1kg', data_trend_u)
        np.save('emg/lizhuo_trend_d_1kg', data_trend_d)
    elif label == '3kg':
        np.save('emg/lizhuo_mean_3kg', data_mean)
        np.save('emg/lizhuo_std_3kg', data_std)
        np.save('emg/lizhuo_trend_u_3kg', data_trend_u)
        np.save('emg/lizhuo_trend_d_3kg', data_trend_d)

    # print(np.max(yt[:, 0, :, :]))
    # print(np.max(yt[:, 1, :, :]))
    # print(np.max(yt[:, 2, :, :]))
    # print(np.max(yt[:, 3, :, :]))
    # print(np.max(yt[:, 4, :, :]))
    plt.show()


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


def plot_result(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c):
    if only_lift is True:
        len1 = target_len
        len2 = emg_lift_len
    else:
        len1 = target_len * 2
        len2 = emg_lift_len * 2
    t = t[num * len1:(num + 1) * len1]
    r = r[:, num * len1:(num + 1) * len1]
    emg = emg[:, num * len1:(num + 1) * len1]
    time = time[num * len1:(num + 1) * len1]
    torque = torque[num * len1:(num + 1) * len1]
    calu_torque = calu_torque[num * len1:(num + 1) * len1]
    emg_std_long = emg_std_long[:, num * len2:(num + 1) * len2]
    emg_mean_long = emg_mean_long[:, num * len2:(num + 1) * len2]
    time_long = resample_by_len(list(time), emg_mean_long.shape[1])
    # if include_TRI is False:
    #     plt.figure(figsize=(6, 7.7))
    #     plt.subplot(411)
    #     plt.plot(time, np.asarray(t)[num * 50:(num + 1) * 50], label='optimization', linewidth=2)
    #     plt.plot(time, torque[num * 50:(num + 1) * 50], label='measured', linewidth=2)
    #     # plt.xlabel('time (s)')
    #     plt.ylabel('torque', weight='bold')
    #     plt.legend()
    #     rmse = np.sqrt(np.sum((np.asarray(t)[num * 50:(num + 1) * 50] - torque[num * 50:(num + 1) * 50]) ** 2) / len(
    #         torque[num * 50:(num + 1) * 50]))
    #     print("torque rmse", num, ":\t", "{:.2f}".format(rmse))
    #
    #     plt.subplot(412)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[0, num * 1000:(num + 1) * 1000] * c[0],
    #                          2 * emg_std_long[0, num * 1000:(num + 1) * 1000] * c[0],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[0, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[0, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[0, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[0, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     # plt.xlabel('time (s)')
    #     plt.ylabel('bic_s_l', weight='bold')
    #     plt.legend()
    #
    #     plt.subplot(413)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[1, num * 1000:(num + 1) * 1000] * c[1],
    #                          2 * emg_std_long[1, num * 1000:(num + 1) * 1000] * c[1],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[1, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[1, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[1, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[1, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     # plt.xlabel('time (s)')
    #     plt.ylabel('brachialis_1_l', weight='bold')
    #     plt.legend()
    #
    #     # plt.figure()
    #     plt.subplot(414)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[2, num * 1000:(num + 1) * 1000] * c[2],
    #                          2 * emg_std_long[2, num * 1000:(num + 1) * 1000] * c[2],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[2, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[2, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[2, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[2, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     plt.xlabel('time (s)', weight='bold')
    #     plt.ylabel('brachiorad_1_l', weight='bold')
    #     plt.legend()
    # else:
    #     plt.figure(figsize=(6, 7.7))
    #     plt.subplot(511)
    #     plt.plot(time, np.asarray(t[num * 50:(num + 1) * 50]), label='optimization', linewidth=2)
    #     plt.plot(time, torque[num * 50:(num + 1) * 50], label='measured', linewidth=2)
    #     # plt.xlabel('time (s)')
    #     plt.ylabel('torque', weight='bold')
    #     plt.legend()
    #     rmse = np.sqrt(np.sum((np.asarray(t[num * 50:(num + 1) * 50]) - torque[num * 50:(num + 1) * 50]) ** 2) / len(
    #         torque[num * 50:(num + 1) * 50]))
    #     print("torque rmse:\t", "{:.2f}".format(rmse))
    #
    #     plt.subplot(512)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[0, num * 1000:(num + 1) * 1000] * c[0],
    #                          2 * emg_std_long[0, num * 1000:(num + 1) * 1000] * c[0],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[0, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[0, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[0, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[0, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     # plt.xlabel('time (s)')
    #     plt.ylabel('bic_s_l', weight='bold')
    #     plt.legend()
    #
    #     plt.subplot(513)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[1, num * 1000:(num + 1) * 1000] * c[1],
    #                          2 * emg_std_long[1, num * 1000:(num + 1) * 1000] * c[1],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[1, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[1, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[1, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[1, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     # plt.xlabel('time (s)')
    #     plt.ylabel('brachialis_1_l', weight='bold')
    #     plt.legend()
    #
    #     # plt.figure()
    #     plt.subplot(514)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[2, num * 1000:(num + 1) * 1000] * c[2],
    #                          2 * emg_std_long[2, num * 1000:(num + 1) * 1000] * c[2],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[2, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[2, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[2, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[2, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     plt.xlabel('time (s)', weight='bold')
    #     plt.ylabel('brachiorad_1_l', weight='bold')
    #
    #     plt.subplot(515)
    #     if plot_distribution is True:
    #         if mvc_is_variable is True:
    #             plt.errorbar(time_long, emg_mean_long[3, num * 1000:(num + 1) * 1000] * c[3],
    #                          2 * emg_std_long[3, num * 1000:(num + 1) * 1000] * c[3],
    #                          label='emg', color='lavender', zorder=1)
    #         else:
    #             plt.errorbar(time_long, emg_mean_long[3, num * 1000:(num + 1) * 1000],
    #                          2 * emg_std_long[3, num * 1000:(num + 1) * 1000],
    #                          label='emg', color='lavender', zorder=1)
    #     else:
    #         plt.plot(time, np.asarray(emg[3, num * 50:(num + 1) * 50]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(r[3, num * 50:(num + 1) * 50]), label='optimization', linewidth=2, zorder=2)
    #     plt.xlabel('time (s)', weight='bold')
    #     plt.ylabel('tri', weight='bold')
    #     plt.legend()
    #
    # plt.figure(figsize=(6, 3.7))
    # # plt.plot(time[num * 50:(num + 1) * 50], calu_torque[num * 50:(num + 1) * 50], label='calculated', linewidth=2)
    # plt.plot(time, torque[num * 50:(num + 1) * 50], label='measured', linewidth=2)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('torque', weight='bold')
    # plt.legend()
    # rmse = np.sqrt(np.sum((np.asarray(calu_torque) - torque) ** 2) / len(torque))
    # print("torque rmse:\t", "{:.2f}".format(rmse))

    if include_TRI is False:
        plt.figure(figsize=(6, 7.7))
        plt.subplot(411)
        plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
        plt.plot(time, torque, label='measured', linewidth=2)
        # plt.xlabel('time (s)')
        plt.ylabel('torque', weight='bold', size=10)
        plt.legend()
        rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
        print("torque rmse", num, ":\t", "{:.2f}".format(rmse))

        plt.subplot(412)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[0, :] * c[0], 2 * emg_std_long[0, :] * c[0], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[0, :], 2 * emg_std_long[0, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[0]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[0, :]), label='optimization', linewidth=2, zorder=2)
        # plt.xlabel('time (s)')
        plt.ylabel('biceps', weight='bold')
        plt.legend()

        plt.subplot(413)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[1, :] * c[1], 2 * emg_std_long[1, :] * c[1], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[1, :], 2.5 * emg_std_long[1, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[1]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[1, :]), label='optimization', linewidth=2, zorder=2)
        # plt.xlabel('time (s)')
        plt.ylabel('brachialis', weight='bold')
        plt.legend()

        # plt.figure()
        plt.subplot(414)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[2, :] * c[2], 2 * emg_std_long[2, :] * c[2], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[2, :], 2 * emg_std_long[2, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[2]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[2, :]), label='optimization', linewidth=2, zorder=2)
        plt.xlabel('time (s)', weight='bold', size=10)
        plt.ylabel('brachiorad', weight='bold')
        plt.legend()
    else:
        plt.figure(figsize=(6, 7.7))
        plt.subplot(511)
        plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
        plt.plot(time, torque, label='measured', linewidth=2)
        # plt.xlabel('time (s)')
        plt.ylabel('torque', weight='bold')
        plt.legend()
        rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
        print("torque rmse:\t", "{:.2f}".format(rmse))

        plt.subplot(512)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[0, :] * c[0], 2 * emg_std_long[0, :] * c[0], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[0, :], 2 * emg_std_long[0, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[0]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[0, :]), label='optimization', linewidth=2, zorder=2)
        # plt.xlabel('time (s)')
        plt.ylabel('bic_s_l', weight='bold')
        plt.legend()

        plt.subplot(513)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[1, :] * c[1], 2 * emg_std_long[1, :] * c[1], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[1, :], 2 * emg_std_long[1, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[1]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[1, :]), label='optimization', linewidth=2, zorder=2)
        # plt.xlabel('time (s)')
        plt.ylabel('brachialis_1_l', weight='bold')
        plt.legend()

        # plt.figure()
        plt.subplot(514)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[2, :] * c[2], 2 * emg_std_long[2, :] * c[2], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[2, :], 2 * emg_std_long[2, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[2]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[2, :]), label='optimization', linewidth=2, zorder=2)
        plt.xlabel('time (s)', weight='bold')
        plt.ylabel('brachiorad_1_l', weight='bold')

        plt.subplot(515)
        if plot_distribution is True:
            if mvc_is_variable is True:
                plt.errorbar(time_long, emg_mean_long[3, :] * c[3], 2 * emg_std_long[3, :] * c[3], label='emg',
                             color='lavender', zorder=1)
            else:
                plt.errorbar(time_long, emg_mean_long[3, :], 2 * emg_std_long[3, :], label='emg', color='lavender',
                             zorder=1)
        else:
            plt.plot(time, np.asarray(emg[3]), label='emg', linewidth=2, zorder=3)
        plt.plot(time, np.asarray(r[3, :]), label='optimization', linewidth=2, zorder=2)
        plt.xlabel('time (s)', weight='bold')
        plt.ylabel('tri', weight='bold')
        plt.legend()
    plt.savefig('train_{}.png'.format(num))

    # plt.figure(figsize=(6, 3.7))
    # plt.plot(time, calu_torque, label='calculated', linewidth=2)
    # plt.plot(time, torque, label='measured', linewidth=2)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('torque', weight='bold')
    # plt.legend()
    rmse = np.sqrt(np.sum((np.asarray(calu_torque) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))


def calurmse(c_r, y_r, emg, arm, torque, time, label):
    if mvc_is_variable is True:
        active_force = emg.T * c_r * y_r
    else:
        active_force = emg.T * y_r
    calu_torque = [sum(active_force[j, :] * arm[:, j]) for j in range(arm.shape[1])]

    plt.figure(figsize=(6, 3.7))
    plt.plot(time, calu_torque, label='calculated', linewidth=2)
    plt.plot(time, torque, label='measured', linewidth=2)
    plt.xlabel('time (s)', weight='bold')
    plt.ylabel('torque', weight='bold')
    plt.legend()
    plt.savefig(label + '_train.png')

    rmse = np.sqrt(np.sum((np.asarray(calu_torque) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))


def test_result(c_r, y_r):
    emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, emg_trend_u1, emg_trend_d1 \
        = read_realted_files(label='chenzui-left-all-6.5kg', idx='18')
    emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, emg_trend_u2, emg_trend_d2 \
        = read_realted_files(label='chenzui-left-all-6.5kg', idx='19')
    emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, emg_trend_u3, emg_trend_d3 \
        = read_realted_files(label='chenzui-left-all-6.5kg', idx='20')
    emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, emg_trend_u4, emg_trend_d4 \
        = read_realted_files(label='chenzui-left-all-6.5kg', idx='21')
    emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, emg_trend_u5, emg_trend_d5 \
        = read_realted_files(label='chenzui-left-all-6.5kg', idx='22')
    # emg_mean6, emg_std6, arm6, fa6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6 \
    #     = read_chenzui_realted_files(label='zhuo-9')
    # test_optimization_emg(0, y_r, torque1, time1, emg_mean_long1, emg_std_long1, arm1, emg1, emg_mean1, emg_std1, emg_trend_u1, emg_trend_d1)
    # test_optimization_emg(1, y_r, torque2, time2, emg_mean_long2, emg_std_long2, arm2, emg2, emg_mean2, emg_std2, emg_trend_u2, emg_trend_d2)
    # test_optimization_emg(2, y_r, torque3, time3, emg_mean_long3, emg_std_long3, arm3, emg3, emg_mean3, emg_std3, emg_trend_u3, emg_trend_d3)
    # test_optimization_emg(3, y_r, torque4, time4, emg_mean_long4, emg_std_long4, arm4, emg4, emg_mean4, emg_std4, emg_trend_u4, emg_trend_d4)
    # test_optimization_emg(4, y_r, torque5, time5, emg_mean_long5, emg_std_long5, arm5, emg5, emg_mean5, emg_std5, emg_trend_u5, emg_trend_d5)
    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, emg_trend_u1, emg_trend_d1 = \
    #     read_realted_files(label='zhuo-right-3kg', idx='3')
    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1 \
    #     = read_realted_files(label='zhuo-2')
    # emg_mean2, emg_std2, arm2, fa2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2 \
    #     = read_realted_files(label='zhuo-3')
    # emg_mean3, emg_std3, arm3, fa3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3 \
    #     = read_realted_files(label='zhuo-9')
    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1 \
    #     = read_chenzui_realted_files(label='5.5kg-2')
    # emg_mean2, emg_std2, arm2, fa2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2 \
    #     = read_chenzui_realted_files(label='5.5kg-6')
    # emg_mean3, emg_std3, arm3, fa3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3 \
    #     = read_chenzui_realted_files(label='5.5kg-7')
    # emg_mean4, emg_std4, arm4, fa4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4 \
    #     = read_chenzui_realted_files(label='15')
    # emg_mean5, emg_std5, arm5, fa5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5 \
    #     = read_chenzui_realted_files(label='17')
    # emg_mean6, emg_std6, arm6, fa6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6 \
    #     = read_chenzui_realted_files(label='18')

    print('-' * 50)
    # calurmse(c_r, y_r, emg1, arm1, torque1, time1, label='zhuo-right-3kg')
    # calurmse(c_r, y_r, emg2, arm2, torque2, time2, label='zhuo-3')
    # calurmse(c_r, y_r, emg3, arm3, torque3, time3, label='zhuo-9')
    calurmse(c_r, y_r, emg1, arm1, torque1, time1, label='6.5kg-1')
    calurmse(c_r, y_r, emg2, arm2, torque2, time2, label='6.5kg-2')
    calurmse(c_r, y_r, emg3, arm3, torque3, time3, label='6.5kg-3')
    calurmse(c_r, y_r, emg4, arm4, torque4, time4, label='6.5kg-4')
    calurmse(c_r, y_r, emg5, arm5, torque5, time5, label='6.5kg-5')
    # calurmse(c_r, y_r, emg6, arm6, torque6, time6, label='3kg-18')


def test_optimization_emg(num, y, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std, emg_trend_u, emg_trend_d):
    # m = gekko.GEKKO(remote=False)
    # x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    # c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    # f = m.Array(m.Var, arm.shape)  # muscle force
    # t = m.Array(m.Var, torque.shape)  # torque
    # # m.Minimize(np.square(x - emg).mean())
    # m.Minimize(np.square(t - torque).mean() + np.square(x).mean())
    # for i in range(arm.shape[0]):
    #     for j in range(arm.shape[1]):
    #         m.Equation(x[i][j] * y[i] == f[i, j])
    #         m.Equation(sum(f[:, j] * arm[:, j]) == t[j])
    #     if mvc_is_variable is True:
    #         m.Equations([c[i] >= 0.01, c[i] <= 5])
    # # m.options.MAX_ITER = 100000
    # m.solve(disp=False)

    time_long = resample_by_len(list(time), emg_mean_long.shape[1])

    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    f = m.Array(m.Var, arm.shape)  # muscle force
    t = m.Array(m.Var, torque.shape)  # torque
    # m.Minimize(np.square(t - torque).mean() + m.sqrt(np.square(x).mean()))
    # m.Minimize(np.square(t - torque).mean() + np.mean(abs(x)))
    m.Minimize(np.square(t - torque).mean())
    # m.Minimize(m.sqrt(((t - torque) ** 2).mean()) + m.sqrt((x ** 2).mean()))
    for i in range(arm.shape[0]):
        for j in range(arm.shape[1]):
            m.Equation(x[i][j] * y[i] == f[i, j])
            m.Equation(sum(f[:, j] * arm[:, j]) == t[j])
            if j < arm.shape[1] - 1:
                m.Equation(x[i][j + 1] <= x[i][j] + target_len / emg_lift_len * 5 * emg_trend_u[i][j])
                m.Equation(x[i][j + 1] >= x[i][j] + target_len / emg_lift_len * 5 * emg_trend_d[i][j])
                # m.Equation(x[i][j + 1] <= x[i][j] + target_len / emg_lift_len * 10 * emg_trend_u[i][j])
                # m.Equation(x[i][j + 1] >= x[i][j] + target_len / emg_lift_len * 10 * emg_trend_d[i][j])
            m.Equation(x[i][j] <= (emg_mean[i][j] + emg_std[i][j] * 2))
            m.Equation(x[i][j] >= (emg_mean[i][j] - emg_std[i][j] * 2))
    # m.options.OTOL = 1  # 容差值
    m.solve(disp=False)

    xx = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = x[i][j].value[0]
            xx[i][j] = xx[i][j] * x[i][j].value[0]
    e = np.ones_like(t)
    for i in range(t.shape[0]):
        e[i] = e[i] * t[i].value[0]
    t = e
    act = np.asarray(xx)
    tor = np.asarray(t)
    # act = np.asarray([x[i].value[0] for i in range(arm.shape)])
    # act = np.asarray([n.value[0] for n in x])
    # tor = np.asarray([t[i].value[0] for i in range(arm.shape[1])])

    # for j in range(arm.shape[1]):
    #     # m = gekko.GEKKO(remote=False)
    #     # x = m.Array(m.Var, arm.shape[0], lb=0, ub=1)  # activation
    #     # c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    #     # f = m.Array(m.Var, arm.shape[0])  # muscle force
    #     # t = m.Var()  # torque
    #     # # t = m.Array(m.Var, 1)  # torque
    #     # # m.Minimize(np.square(t - torque[j]) + np.square(x).sum())
    #     # m.Minimize(np.square(t - torque[j]))
    #     # # m.Minimize(np.square(x).sum())
    #     for i in range(arm.shape[0]):
    #         m.Equation(x[i] * y[i] == f[i])
    #         if j < arm.shape[1] - 1:
    #             m.Equation(x[i][j + 1] <= x[i][j] + 0.4 * emg_trend_u[i][j])
    #             m.Equation(x[i][j + 1] >= x[i][j] + 0.4 * emg_trend_d[i][j])
    #     m.Equation(sum(f * arm[:, j]) == t)
    #     # m.Equation(sum(f * arm[:, j]) == torque[j])
    #     m.solve(disp=False)
    #
    #     tt = 1
    #     xx = np.ones_like(x)
    #     tt = tt * t.value[0]
    #     for i in range(x.shape[0]):
    #         xx[i] = xx[i] * x[i].value[0]
    #     act.append(xx)
    #     tor.append(tt)
    # act = np.asarray(act)
    # tor = np.asarray(tor)

    # plt.figure(figsize=(6, 7.7))
    # plt.subplot(411)
    # plt.plot(time, tor, label='optimization', linewidth=2)
    # plt.plot(time, torque, label='measured', linewidth=2)
    # plt.ylabel('torque', weight='bold', size=10)
    # plt.legend()
    # rmse = np.sqrt(np.sum((tor - torque) ** 2) / len(torque))
    # print("torque rmse", num, ":\t", "{:.2f}".format(rmse))
    #
    # plt.subplot(412)
    # if test_plot_distribution is True:
    #     if mvc_is_variable is True:
    #         plt.errorbar(time_long, emg_mean_long[0, :] * c[0], 2 * emg_std_long[0, :] * c[0], label='emg',
    #                      color='lavender', zorder=1)
    #     else:
    #         plt.errorbar(time_long, emg_mean_long[0, :], 2 * emg_std_long[0, :], label='emg', color='lavender',
    #                      zorder=1)
    # else:
    #     plt.plot(time, np.asarray(emg[0]), label='emg', linewidth=2, zorder=3)
    # plt.plot(time, np.asarray(act[:, 0]), label='optimization', linewidth=2, zorder=2)
    # plt.ylabel('bic_s_l', weight='bold')
    # plt.legend()
    #
    # plt.subplot(413)
    # if test_plot_distribution is True:
    #     if mvc_is_variable is True:
    #         plt.errorbar(time_long, emg_mean_long[1, :] * c[1], 2 * emg_std_long[1, :] * c[1], label='emg',
    #                      color='lavender', zorder=1)
    #     else:
    #         plt.errorbar(time_long, emg_mean_long[1, :], 2 * emg_std_long[1, :], label='emg', color='lavender',
    #                      zorder=1)
    # else:
    #     plt.plot(time, np.asarray(emg[1]), label='emg', linewidth=2, zorder=3)
    # plt.plot(time, np.asarray(act[:, 1]), label='optimization', linewidth=2, zorder=2)
    # # plt.xlabel('time (s)')
    # plt.ylabel('brachialis_1_l', weight='bold')
    # plt.legend()
    #
    # # plt.figure()
    # plt.subplot(414)
    # if test_plot_distribution is True:
    #     if mvc_is_variable is True:
    #         plt.errorbar(time_long, emg_mean_long[2, :] * c[2], 2 * emg_std_long[2, :] * c[2], label='emg',
    #                      color='lavender', zorder=1)
    #     else:
    #         plt.errorbar(time_long, emg_mean_long[2, :], 2 * emg_std_long[2, :], label='emg', color='lavender',
    #                      zorder=1)
    # else:
    #     plt.plot(time, np.asarray(emg[2]), label='emg', linewidth=2, zorder=3)
    # plt.plot(time, np.asarray(act[:, 2]), label='optimization', linewidth=2, zorder=2)
    # plt.xlabel('time (s)', weight='bold', size=10)
    # plt.ylabel('brachiorad_1_l', weight='bold')
    # plt.legend()

    plt.figure(figsize=(6, 7.7))
    plt.subplot(411)
    plt.plot(time, tor, label='optimization', linewidth=2)
    plt.plot(time, torque, label='measured', linewidth=2)
    plt.ylabel('torque', weight='bold', size=10)
    plt.legend()
    rmse = np.sqrt(np.sum((tor - torque) ** 2) / len(torque))
    print("torque rmse", ":\t", "{:.2f}".format(rmse))

    plt.subplot(412)
    plt.errorbar(time_long, emg_mean_long[0, :], 2 * emg_std_long[0, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(act[0, :]), label='optimization', linewidth=2, zorder=2)
    plt.plot(time, np.asarray(emg[0, :]), label='emg', linewidth=2, zorder=3)
    plt.ylabel('bic_s_l', weight='bold')
    rmse = np.sqrt(np.sum((act[0, :] - emg[0, :]) ** 2) / len(emg[0, :]))
    print("torque rmse", num, ":\t", "{:.2f}".format(rmse))
    plt.legend()

    plt.subplot(413)
    plt.errorbar(time_long, emg_mean_long[1, :], 2 * emg_std_long[1, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(act[1, :]), label='optimization', linewidth=2, zorder=2)
    plt.plot(time, np.asarray(emg[1, :]), label='emg', linewidth=2, zorder=3)
    # plt.xlabel('time (s)')
    plt.ylabel('brachialis_1_l', weight='bold')
    rmse = np.sqrt(np.sum((act[1, :] - emg[1, :]) ** 2) / len(emg[1, :]))
    print("torque rmse", num, ":\t", "{:.2f}".format(rmse))
    plt.legend()

    plt.subplot(414)
    plt.errorbar(time_long, emg_mean_long[2, :], 2 * emg_std_long[2, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(act[2, :]), label='optimization', linewidth=2, zorder=2)
    plt.plot(time, np.asarray(emg[2, :]), label='emg', linewidth=2, zorder=3)
    plt.xlabel('time (s)', weight='bold', size=10)
    plt.ylabel('brachiorad_1_l', weight='bold')
    rmse = np.sqrt(np.sum((act[2, :] - emg[2, :]) ** 2) / len(emg[2, :]))
    print("torque rmse", num, ":\t", "{:.2f}".format(rmse))
    plt.legend()
    plt.savefig('{}_test.png'.format(num))


def read_groups_files(label='zhuo-right-3kg'):
    if label == 'zhuo-right-3kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='zhuo-right-3kg', idx='2')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='zhuo-right-3kg', idx='3')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='zhuo-right-3kg', idx='5')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='zhuo-right-3kg', idx='6')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='zhuo-right-3kg', idx='8')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'zhuo-left-1kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='zhuo-left-1kg', idx='3')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='zhuo-left-1kg', idx='4')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='zhuo-left-1kg', idx='6')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='zhuo-left-1kg', idx='7')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='zhuo-left-1kg', idx='9')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-5.5kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-5.5kg', idx='2')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-5.5kg', idx='3')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-5.5kg', idx='4')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-5.5kg', idx='5')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-5.5kg', idx='6')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-3kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-3kg', idx='13')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-3kg', idx='14')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-3kg', idx='15')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-3kg', idx='17')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-3kg', idx='18')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-all-2.5kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='5')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='6')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='7')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='8')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='9')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-all-3kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='10')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='11')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='12')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='13')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='14')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-all-4kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-4kg', idx='15')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-4kg', idx='16')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-4kg', idx='17')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3), axis=1)
        arm = np.concatenate((arm1, arm2, arm3), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3), axis=0)
        time = np.concatenate((time1, time2, time3), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3), axis=1)
        emg = np.concatenate((emg1, emg2, emg3), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-all-6.5kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='18')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='19')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='20')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='21')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='22')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d

    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1 \
    #     = read_realted_files(label='zhuo-4')
    # emg_mean2, emg_std2, arm2, fa2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2 \
    #     = read_realted_files(label='zhuo-6')
    # emg_mean3, emg_std3, arm3, fa3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3 \
    #     = read_realted_files(label='zhuo-7')
    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1 \
    #     = read_chenzui_realted_files(label='11')
    # emg_mean2, emg_std2, arm2, fa2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2 \
    #     = read_chenzui_realted_files(label='13')
    # emg_mean3, emg_std3, arm3, fa3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3 \
    #     = read_chenzui_realted_files(label='14')

    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1 \
    #     = read_chenzui_realted_files(label='5.5kg-3')
    # emg_mean2, emg_std2, arm2, fa2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2 \
    #     = read_chenzui_realted_files(label='5.5kg-4')
    # emg_mean3, emg_std3, arm3, fa3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3 \
    #     = read_chenzui_realted_files(label='5.5kg-5')

    # emg_mean1, emg_std1, arm1, fa1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1 \
    #     = read_chenzui_realted_files(label='11')
    # emg_mean2, emg_std2, arm2, fa2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2 \
    #     = read_chenzui_realted_files(label='13')
    # emg_mean3, emg_std3, arm3, fa3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3 \
    #     = read_chenzui_realted_files(label='14')
    # emg_mean4, emg_std4, arm4, fa4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4 \
    #     = read_chenzui_realted_files(label='5.5kg-2')
    # emg_mean5, emg_std5, arm5, fa5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5 \
    #     = read_chenzui_realted_files(label='5.5kg-3')
    # emg_mean6, emg_std6, arm6, fa6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6 \
    #     = read_chenzui_realted_files(label='5.5kg-4')
    #
    # emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5, emg_mean6), axis=1)
    # emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5, emg_std6), axis=1)
    # arm = np.concatenate((arm1, arm2, arm3, arm4, arm5, arm6), axis=1)
    # # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
    # torque = np.concatenate((torque1, torque2, torque3, torque4, torque5, torque6), axis=0)
    # time = np.concatenate((time1, time2, time3, time4, time5, time6), axis=0)
    # emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4,
    #                                 emg_mean_long5, emg_mean_long6), axis=1)
    # emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4,
    #                                emg_std_long5, emg_std_long6), axis=1)
    # emg = np.concatenate((emg1, emg2, emg3, emg4, emg5, emg6), axis=1)
    # time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5, time_emg6), axis=0)

    # emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3), axis=1)
    # emg_std = np.concatenate((emg_std1, emg_std2, emg_std3), axis=1)
    # arm = np.concatenate((arm1, arm2, arm3), axis=1)
    # fa = np.concatenate((fa1, fa2, fa3), axis=1)
    # torque = np.concatenate((torque1, torque2, torque3), axis=0)
    # time = np.concatenate((time1, time2, time3), axis=0)
    # emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3), axis=1)
    # emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3), axis=1)
    # emg = np.concatenate((emg1, emg2, emg3), axis=1)
    # time_emg = np.concatenate((time_emg1, time_emg2, time_emg3), axis=0)

    # return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    # return emg_mean, emg_std, arm, fa, torque, time, emg_mean_long, emg_std_long, emg, time_emg


if __name__ == '__main__':
    # calculate_emg_distribution()
    calculate_chenzui_emg_distribution(label='2.5kg-all')
    # calculate_lizhuo_emg_distribution(label='1kg')
    # calculate_torque()
    # print(calc_fal(0.1, 'bic_s_l'))
    # emg_mean, emg_std, arm, fa, torque, time, emg_mean_long, emg_std_long, emg, time_emg = read_realted_files(
    #     label='zhuo-7')
    # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d\
    #     = read_realted_files(label='zhuo-right-3kg', idx='2')
    # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    #     = read_realted_files(label='chenzui-left-3kg', idx='13')
    emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
        = read_groups_files('chenzui-left-all-6.5kg')
    iso = np.asarray(iso)
    # fa = np.asarray(fa)
    arm = np.asarray(arm)
    time = np.asarray(time)
    torque = np.asarray(torque)
    emg_std = np.asarray(emg_std)
    emg_mean = np.asarray(emg_mean)
    emg_trend_u = np.asarray(emg_trend_u)
    emg_trend_d = np.asarray(emg_trend_d)
    emg = np.asarray(emg)
    time_emg = np.asarray(time_emg)

    # action_idx = 0
    # fa = fa[:, action_idx, :]
    # arm = arm[:, action_idx, :]
    # time = time[action_idx, :]
    # time_long = resample_by_len(list(time), emg_mean_long.shape[1])

    # iso_copy = [346, 628, 60]
    # fr = np.asarray(
    #     [emg[0][action_idx] * iso_copy[0], emg[1][action_idx] * iso_copy[1], emg[2][action_idx] * iso_copy[2]])
    # torque = np.array([sum(fr[:, i] * arm[:, i]) for i in range(arm.shape[1])])
    # to = torque
    # noise = np.random.normal(0, 1, to.shape)
    # torque = to + noise
    # torque = torque[action_idx, :]
    # torque = torque - torque[0]

    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    y = m.Array(m.Var, len(muscle_idx), lb=0)  # maximum isometric force
    c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    f = m.Array(m.Var, arm.shape)  # muscle force
    t = m.Array(m.Var, torque.shape)  # torque
    # m.Minimize(np.square(x - emg).mean())
    m.Minimize(np.square(t - torque).mean())
    for i in range(arm.shape[0]):
        for j in range(arm.shape[1]):
            m.Equation(x[i][j] * y[i] == f[i, j])
            # m.Equation(x[i][j] * y[i] * fa[i][j] == f[i, j])
            m.Equation(sum(f[:, j] * arm[:, j]) == t[j])
            if mvc_is_variable is True:
                m.Equation(x[i][j] <= (emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2))
                m.Equation(x[i][j] >= (emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2))
                m.Equation((emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2) <= 1)
                # m.Equation((emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2) >= 0)
            else:
                # if j < arm.shape[1] - 1:
                #     # m.Equation(x[i][j + 1] <= x[i][j] * emg_trend_u[i][j])
                #     # m.Equation(x[i][j + 1] >= x[i][j] * emg_trend_d[i][j])
                #     m.Equation(x[i][j + 1] <= x[i][j] + 0.4 * emg_trend_u[i][j])
                #     m.Equation(x[i][j + 1] >= x[i][j] + 0.4 * emg_trend_d[i][j])
                m.Equation(x[i][j] <= emg[i][j] * 1.20)
                m.Equation(x[i][j] >= emg[i][j] * 0.80)
                # m.Equation(x[i][j] <= (emg_mean[i][j] + emg_std[i][j] * 2))
                # m.Equation(x[i][j] >= (emg_mean[i][j] - emg_std[i][j] * 2))
        # m.Equations([y[i] >= 3 * iso1[i], y[i] <= 3 * iso2[i]])
        # m.Equations([y[i] >= iso1[i], y[i] <= iso2[i]])
        m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 50 * iso[i]])
        # m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 3 * iso[i]])
        if mvc_is_variable is True:
            m.Equations([c[i] >= 0.01, c[i] <= 5])
    # m.options.MAX_ITER = 10000
    m.solve(disp=False)
    # print(x)
    # print(y)
    for i in range(y.size):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i].value[0]))
    if mvc_is_variable is True:
        print('-' * 50)
        for i in range(c.size):
            print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(c)[i].value[0]))

    r = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i][j] = r[i][j] * x[i][j].value[0]
    e = np.ones_like(t)
    for i in range(t.shape[0]):
        e[i] = e[i] * t[i].value[0]
    t = e

    c_r = [(np.asarray(c)[i].value[0]) for i in range(c.size)]
    y_r = [(np.asarray(y)[i].value[0]) for i in range(y.size)]  # maximum isometric force
    if mvc_is_variable is True:
        active_force = emg.T * c_r * y_r
    else:
        active_force = emg.T * y_r
        # active_force = fa.T * emg.T * y_r
    calu_torque = [sum(active_force[j, :] * arm[:, j]) for j in range(arm.shape[1])]

    print('-' * 50)
    rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    plot_result(0, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    plot_result(1, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    plot_result(2, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    plot_result(3, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    plot_result(4, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    # plot_result(5, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    #
    # test_result(c_r, y_r)
    #
    # print('-' * 50)
    # y_r = np.array([436.65, 157.00, 15])
    # y_r = np.array([134.57, 846.82, 150.00])
    # y_r = np.array([8126.39, 15700.00, 485.59])
    # y_r = np.array([1484.95, 242.86, 445.27])
    # emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, emg_trend_u1, emg_trend_d1 \
    #     = read_realted_files(label='zhuo-left-1kg', idx='3')
    # emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, emg_trend_u2, emg_trend_d2 \
    #     = read_realted_files(label='zhuo-left-1kg', idx='4')
    # emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, emg_trend_u3, emg_trend_d3 \
    #     = read_realted_files(label='zhuo-left-1kg', idx='6')
    # emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, emg_trend_u4, emg_trend_d4 \
    #     = read_realted_files(label='zhuo-left-1kg', idx='7')
    # emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, emg_trend_u5, emg_trend_d5 \
    #     = read_realted_files(label='zhuo-left-1kg', idx='9')
    # # emg_mean6, emg_std6, arm6, fa6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6 \
    # #     = read_chenzui_realted_files(label='zhuo-9')
    # test_optimization_emg(0, y_r, torque1, time1, emg_mean_long1, emg_std_long1, arm1, emg1, emg_mean1, emg_std1, emg_trend_u1, emg_trend_d1)
    # test_optimization_emg(1, y_r, torque2, time2, emg_mean_long2, emg_std_long2, arm2, emg2, emg_mean2, emg_std2, emg_trend_u2, emg_trend_d2)
    # test_optimization_emg(2, y_r, torque3, time3, emg_mean_long3, emg_std_long3, arm3, emg3, emg_mean3, emg_std3, emg_trend_u3, emg_trend_d3)
    # test_optimization_emg(3, y_r, torque4, time4, emg_mean_long4, emg_std_long4, arm4, emg4, emg_mean4, emg_std4, emg_trend_u4, emg_trend_d4)
    # test_optimization_emg(4, y_r, torque5, time5, emg_mean_long5, emg_std_long5, arm5, emg5, emg_mean5, emg_std5, emg_trend_u5, emg_trend_d5)
    # # test_optimization_emg(5, y_r, torque6, time6, emg_mean_long6, emg_std_long6, arm6, emg6, emg_mean6, emg_std6, emg_trend_u6, emg_trend_d6)

    # plt.figure(figsize=(6, 7.7))
    # plt.subplot(311)
    # plt.plot(time, np.asarray(emg[0]), label='emg', linewidth=2, zorder=3)
    # plt.ylabel('bic_s_l', weight='bold')
    # plt.legend()
    #
    # plt.subplot(312)
    # plt.plot(time, np.asarray(emg[1]), label='emg', linewidth=2, zorder=3)
    # # plt.xlabel('time (s)')
    # plt.ylabel('brachialis_1_l', weight='bold')
    # plt.legend()
    #
    # plt.subplot(313)
    # plt.plot(time, np.asarray(emg[2]), label='emg', linewidth=2, zorder=3)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('brachiorad_1_l', weight='bold')
    # plt.legend()

    # plt.subplot(615)
    # plt.errorbar(time_long, emg_mean_long[3, :], 2 * emg_std_long[3, :], label='emg', color='lavender', zorder=1)
    # plt.plot(time, np.asarray(r[3, :]), label='optimization', linewidth=2, zorder=2)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('TRIlong', weight='bold')
    # plt.legend()
    #
    # plt.subplot(616)
    # plt.errorbar(time_long, emg_mean_long[4, :], 2 * emg_std_long[4, :], label='emg', color='lavender', zorder=1)
    # plt.plot(time, np.asarray(r[4, :]), label='optimization', linewidth=2, zorder=2)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('TRIlat', weight='bold')
    # plt.legend()
    plt.show()
