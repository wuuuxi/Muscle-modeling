import datetime

import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
# import cvxpy as cp
import numpy as np
import pandas as pd
# from mpmath import diff
import scipy
# import gekko
from scipy.optimize import leastsq

from read_files import *
from basic import *


def read_angle_files(label):
    fs = 2000
    JA = 'Xsens_jointangle_q-'
    people = 'chenzui'
    file_folder = 'files/bench press/chenzui/'
    if label == 'bp-chenzui-left-4kg':
        emg = np.load(file_folder + '4' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '4' + '.xlsx')
        t_delta_joi = 10.266
        t_delta_emg = 5.916
        time_range = [21.999, 67.296]
    elif label == 'bp-chenzui-left-5.5kg':
        emg = np.load(file_folder + '5.5' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '5.5' + '.xlsx')
        t_delta_emg = 5.655
        t_delta_joi = 6.467
        time_range = [16.266, 77.463]
    elif label == 'bp-chenzui-left-6.5kg':
        emg = np.load(file_folder + '6.5' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '6.5' + '.xlsx')
        t_delta_emg = 6.8905
        t_delta_joi = 5.833
        time_range = [13.533, 61.164]
    elif label == 'bp-chenzui-left-7kg':
        emg = np.load(file_folder + '7' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '7' + '.xlsx')
        t_delta_emg = 5.6485
        t_delta_joi = 6.1
        time_range = [15.566, 56.731]
    elif label == 'bp-chenzui-left-9.5kg':
        emg = np.load(file_folder + '9.5' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '9.5' + '.xlsx')
        t_delta_emg = 6.628
        t_delta_joi = 5.633
        time_range = [16.699, 56.83]
    else:
        print('No such label!')
        return 0

    time_angle = joint_angle['time']
    angle1 = joint_angle['arm_flex_l']
    angle2 = joint_angle['elbow_flex_l']

    idxs = find_nearest_idx(time_angle, time_range[0])
    idxe = find_nearest_idx(time_angle, time_range[-1])
    time_angle = time_angle[idxs:idxe]
    angle1 = np.asarray(angle1[idxs:idxe])
    angle2 = np.asarray(angle2[idxs:idxe])

    if sport_label == 'bench_press':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people)
        [emg_TRI, t2] = emg_rectification(emg[:, 2], fs, 'TRI', people)
        [emg_ANT, t3] = emg_rectification(emg[:, 3], fs, 'ANT', people)
        [emg_POS, t4] = emg_rectification(emg[:, 4], fs, 'POS', people)
        [emg_PEC, t5] = emg_rectification(emg[:, 5], fs, 'PEC', people)
        [emg_LAT, t6] = emg_rectification(emg[:, 6], fs, 'LAT', people)
    t1 = t1 - t_delta_emg + t_delta_joi

    emg = [([]) for _ in range(len(muscle_idx))]
    time_emg = []
    for t in time_angle:
        idx = find_nearest_idx(t1, t)
        time_emg.append(t1[idx])
        emg[0].append(emg_BIC[idx])
        emg[1].append(emg_TRI[idx])
        emg[2].append(emg_ANT[idx])
        emg[3].append(emg_POS[idx])
        emg[4].append(emg_PEC[idx])
        emg[5].append(emg_LAT[idx])
    emg = np.asarray(emg).squeeze()

    output = {
        'angle1': angle1,
        'angle2': angle2,
        'emg': emg
    }
    return output


def func(x1, x2, w1, w2, p):
    a, b, c, d, e, f = p
    # result = a + b * x2 + c * x2 ** 2 + d * x2 ** 3
    # result = b * x1 + c * x2
    result = a + b * x1 * w1 + c * x2 * w2 + d * x1 ** 2 + e * x1 * x2 + f * x2 ** 2
    # result = a + d * x1 ** 2 + e * x1 * x2 + f * x2 ** 2
    # result = (a + b * x1 * w1 + c * x2 * w2 + d * x1 ** 2 + e * x1 * x2 + f * x2 ** 2 + g * x1 ** 3 + h * x1 ** 2 * x2
    #           + i * x1 * x2 ** 2 + j * x2 ** 3)
    return result


def angle_function(angle1, angle2, emg, w1=1, w2=1):
    def f_err(p, y, x1, x2, w1, w2):
        return y - func(x1, x2, w1, w2, p)

    p_prior = np.ones(6)
    c = leastsq(f_err, p_prior, args=(emg, angle1, angle2, w1, w2))
    # print(c)
    return c[0], func


def new_func(x1, x2, i):
    # x1: shoulder flextion/extension angle
    # x2: elbow flextion/extension angle
    # m1: parameters of biceps brachii
    # m2: parameters of triceps brachii
    # m3: parameters of anterior deltoid
    # m4: parameters of medius deltoid
    # m5: parameters of pectoralis major
    # m6: parameters of latissimus dorsi
    m1 = [-5.24599425e-02, 1.12795329e-03, 2.14057582e-04, -1.43743556e-06, -4.58787941e-06, 4.34317021e-06]
    m2 = [-5.12585358e-02, 1.26214409e-03, 1.25807236e-04, -1.90665510e-06, -4.55702386e-06, 7.04129111e-06]
    m3 = [-7.04384361e-01, 1.28943856e-02, 3.46272097e-03, -1.70514619e-05, -4.36517607e-05, 6.22048364e-05]
    m4 = [-4.51244496e-03, 2.71907096e-04, -4.78071091e-05, -6.90783105e-07, -1.06911225e-06, 2.52637229e-06]
    m5 = [-3.99286054e-01, 6.88321672e-03, 2.72494923e-03, -5.27667252e-06, -2.04809474e-05, 2.24693104e-05]
    m6 = [-1.37267012e-03, 1.33998386e-04, 1.08466561e-04, -4.53382853e-08, -1.18574291e-07, 6.68846871e-08]
    m = [m1, m2, m3, m4, m5, m6]
    # w1 and w2 is related to the people and the motion.
    # w1 * x1 + w2 * x2 is around 100.
    w1 = 1
    w2 = 1

    act = m[i][0] + m[i][1] * x1 * w1 + m[i][2] * x2 * w2 + m[i][3] * x1 ** 2 + m[i][4] * x1 * x2 + m[i][5] * x2 ** 2
    return act


def benchpress_chenzui():
    # o = read_groups_files('bp-chenzui-left-4.1kg')
    o1 = read_angle_files('bp-chenzui-left-4kg')
    o2 = read_angle_files('bp-chenzui-left-5.5kg')
    o3 = read_angle_files('bp-chenzui-left-6.5kg')
    o4 = read_angle_files('bp-chenzui-left-7kg')
    o5 = read_angle_files('bp-chenzui-left-9.5kg')
    plt.figure()
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        # ax = plt.axes(projection="3d")
        ax.scatter(o1['angle1'], o1['angle2'], o1['emg'][i, :], s=1, color='blue')
        ax.scatter(o2['angle1'], o2['angle2'], o2['emg'][i, :], s=1, color='green')
        ax.scatter(o3['angle1'], o3['angle2'], o3['emg'][i, :], s=1, color='black')
        ax.scatter(o4['angle1'], o4['angle2'], o4['emg'][i, :], s=1, color='orange')
        ax.scatter(o5['angle1'], o5['angle2'], o5['emg'][i, :], s=1, color='red')
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.scatter(o1['angle1'], o1['emg'][i, :], s=1, color='blue')
        plt.scatter(o2['angle1'], o2['emg'][i, :], s=1, color='green')
        plt.scatter(o3['angle1'], o3['emg'][i, :], s=1, color='black')
        plt.scatter(o4['angle1'], o4['emg'][i, :], s=1, color='orange')
        plt.scatter(o5['angle1'], o5['emg'][i, :], s=1, color='red')
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.scatter(o5['angle2'], o5['emg'][i, :], s=1, color='red')
        plt.scatter(o4['angle2'], o4['emg'][i, :], s=1, color='orange')
        plt.scatter(o3['angle2'], o3['emg'][i, :], s=1, color='black')
        plt.scatter(o2['angle2'], o2['emg'][i, :], s=1, color='green')
        plt.scatter(o1['angle2'], o1['emg'][i, :], s=1, color='blue')
    # plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.scatter(o5['angle2'], o5['emg'][i, :], s=1, color='red')

    plt.figure()
    kk = 0
    w1 = [0.99108757, 1.07314675, 0.77712980, 0.92821843, 1.19833492]
    w2 = [1.06774720, 1.11488997, 0.76496521, 0.85938709, 0.90138082]
    o = [o1, o2, o3, o4, o5]
    for j in range(len(o)):
        # plt.figure()
        for i in range(6):
            c, function = angle_function(o[j]['angle1'], o[j]['angle2'], o[j]['emg'][i, :], 1, 1)
            plt.subplot(2, 6, i + 1)
            # plt.scatter(o['angle2'], o['emg'][i, :], s=1, color='red')
            plt.plot(o[j]['angle1'], function(o[j]['angle1'], o[j]['angle2'], w1[j], w2[j], c), '--', label=kk + 1)
            plt.subplot(2, 6, i + 7)
            # plt.scatter(o['angle2'], o['emg'][i, :], s=1, color='red')
            plt.plot(o[j]['angle2'], function(o[j]['angle1'], o[j]['angle2'], w1[j], w2[j], c), '--', label=kk + 1)
            # print(c)
    plt.legend()

    plt.figure()
    leng = [len(o1['angle1']), len(o2['angle1']), len(o3['angle1']), len(o4['angle1']), len(o5['angle1'])]
    angle1 = np.concatenate([o1['angle1'], o2['angle1'], o3['angle1'], o4['angle1'], o5['angle1']], axis=0)
    angle2 = np.concatenate([o1['angle2'], o2['angle2'], o3['angle2'], o4['angle2'], o5['angle2']], axis=0)
    emg = np.concatenate([o1['emg'], o2['emg'], o3['emg'], o4['emg'], o5['emg']], axis=1)
    a = np.concatenate([np.ones(leng[0]) * 0.99108757, np.ones(leng[1]) * 1.07314675, np.ones(leng[2]) * 0.7771298,
                        np.ones(leng[3]) * 0.92821843, np.ones(leng[4]) * 1.19833492])
    b = np.concatenate([np.ones(leng[0]) * 1.06774720, np.ones(leng[1]) * 1.11488997, np.ones(leng[2]) * 0.76496521,
                        np.ones(leng[3]) * 0.85938709, np.ones(leng[4]) * 0.90138082])
    for i in range(6):
        c, function = angle_function(angle1, angle2, emg[i, :], a, b)
        plt.subplot(2, 6, i + 1)
        plt.plot(angle1, function(angle1, angle2, a, b, c), '--', label=kk)
        plt.subplot(2, 6, i + 7)
        plt.plot(angle2, function(angle1, angle2, a, b, c), '--', label=kk)

    plt.figure()
    for i in range(6):
        c, function = angle_function(angle1, angle2, emg[i, :], a, b)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ax.scatter(angle1, angle2, emg[i, :], s=1, color='blue')
        ax.scatter(angle1, angle2, func(angle1, angle2, a, b, c), s=1, color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(angle1, angle2, emg[i, :], a, b)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ax.scatter(angle1, angle2, func(angle1, angle2, a, b, c), s=1, color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(angle1, angle2, emg[i, :], a, b)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ang1 = np.linspace(-50, 100, 500)
        ang2 = np.linspace(20, 120, 500)
        X, Y = np.meshgrid(ang1, ang2)
        # X = X.flatten()
        # Y = Y.flatten()
        ax.plot_surface(X, Y, func(X, Y, 1, 1, c), cmap='rainbow')

    plt.figure()
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ang1 = np.linspace(-50, 100, 500)
        ang2 = np.linspace(20, 120, 500)
        X, Y = np.meshgrid(ang1, ang2)
        # X = X.flatten()
        # Y = Y.flatten()
        ax.plot_surface(X, Y, new_func(X, Y, i), cmap='rainbow')

    plt.figure()
    for i in range(6):
        c, function = angle_function(angle1, angle2, emg[i, :], a, b)
        print(c)
        ax = plt.subplot(2, 3, i + 1)
        ang1 = np.linspace(-50, 100, 500)
        ang2 = np.linspace(20, 120, 500)
        ax.plot(ang2, func(30, ang2, 1, 1, c))


def emg_progressing_squat(emg, fs=1000, people=None, left=True):
    if left is True:
        [emg_BIC, t1] = emg_rectification(emg['LTA'], fs, 'LTA', people)
        [emg_TRI, t2] = emg_rectification(emg['LGL'], fs, 'LGL', people)
        [emg_ANT, t3] = emg_rectification(emg['LVL'], fs, 'LVL', people)
        [emg_POS, t4] = emg_rectification(emg['LRF'], fs, 'LRF', people)
        [emg_PEC, t5] = emg_rectification(emg['LST'], fs, 'LST', people)
        [emg_LAT, t6] = emg_rectification(emg['LBF'], fs, 'LBF', people)
    else:
        [emg_BIC, t1] = emg_rectification(emg['RTA'], fs, 'LTA', people)
        [emg_TRI, t2] = emg_rectification(emg['RGL'], fs, 'LGL', people)
        [emg_ANT, t3] = emg_rectification(emg['RVL'], fs, 'LVL', people)
        [emg_POS, t4] = emg_rectification(emg['RRF'], fs, 'LRF', people)
        [emg_PEC, t5] = emg_rectification(emg['RST'], fs, 'LST', people)
        [emg_LAT, t6] = emg_rectification(emg['RBF'], fs, 'LBF', people)
    emg_list = np.asarray([emg_BIC, emg_TRI, emg_ANT, emg_POS, emg_PEC, emg_LAT])
    t_list = np.asarray([t1, t2, t3, t4, t5, t6])
    return [emg_list, t_list[:, :emg_list.shape[1]]]


def joint_processing_squat(q1, Fs_q, t_emg, emg_raw, left=True):
    if left is True:
        q_1 = np.asarray(q1['Left Knee Flexion/Extension'])
        q_2 = np.asarray(q1['Left Hip Flexion/Extension'])
    else:
        q_1 = np.asarray(q1['Right Knee Flexion/Extension'])
        q_2 = np.asarray(q1['Right Hip Flexion/Extension'])
    q_time = np.arange(0, q_1.size / Fs_q, 1 / Fs_q)
    time = []
    emg = [([]) for _ in range(6)]
    q_knee = []
    q_hip = []
    for i in range(len(q_time)):
        if q_time[i] <= q_time[-1] and q_time[i] <= t_emg[0, -1]:
            idx_t = find_nearest_idx(t_emg[0, :], q_time[i])
            time.append(t_emg[0, idx_t])
            for j in range(6):
                emg[j].append(emg_raw[j, idx_t])
            q_knee.append(q_1[i])
            q_hip.append(q_2[i])
    time = np.asarray(time)
    emg = np.asarray(emg)
    q_knee = np.asarray(q_knee)
    q_hip = np.asarray(q_hip)
    return time, emg, q_knee, q_hip


def squat_yuetian():
    Fs_q = 60
    Fs_e = 1000
    left_label = False
    file_folder = 'files/squat/'
    e1 = pd.read_excel(file_folder + 'emg/Changqiu_Squate_Noload_test1_2024_2_1_13_17_18.xlsx')
    e2 = pd.read_excel(file_folder + 'emg/Changqiu_Squate_20kg_test1_2024_2_1_13_20_58.xlsx')
    e3 = pd.read_excel(file_folder + 'emg/Changqiu_Squate_25kg_test1_2024_2_1_13_26_13-LAPTOP-S5BT2C0K.xlsx')
    e4 = pd.read_excel(file_folder + 'emg/Changqiu_Squate_30kg_test1_2024_2_1_13_33_26.xlsx')
    q1 = pd.read_excel(file_folder + 'xlsx/2.xlsx')
    q2 = pd.read_excel(file_folder + 'xlsx/3.xlsx')
    q3 = pd.read_excel(file_folder + 'xlsx/4.xlsx')
    q4 = pd.read_excel(file_folder + 'xlsx/5.xlsx')
    [e1, t_emg1] = emg_progressing_squat(e1, Fs_e, 'yuetian', left_label)
    [e2, t_emg2] = emg_progressing_squat(e2, Fs_e, 'yuetian', left_label)
    [e3, t_emg3] = emg_progressing_squat(e3, Fs_e, 'yuetian', left_label)
    [e4, t_emg4] = emg_progressing_squat(e4, Fs_e, 'yuetian', left_label)
    time1, emg1, q_knee1, q_hip1 = joint_processing_squat(q1, Fs_q, t_emg1, e1, left_label)
    time2, emg2, q_knee2, q_hip2 = joint_processing_squat(q2, Fs_q, t_emg2, e2, left_label)
    time3, emg3, q_knee3, q_hip3 = joint_processing_squat(q3, Fs_q, t_emg3, e3, left_label)
    time4, emg4, q_knee4, q_hip4 = joint_processing_squat(q4, Fs_q, t_emg4, e4, left_label)
    time = np.concatenate((time1, time2, time3, time4), axis=0)
    emg = np.concatenate((emg1, emg2, emg3, emg4), axis=1)
    q_knee = np.concatenate((q_knee1, q_knee2, q_knee3, q_knee4), axis=0)
    q_hip = np.concatenate((q_hip1, q_hip2, q_hip3, q_hip4), axis=0)

    # c, function = angle_function(q_knee, q_hip, emg[0], 1, 1)
    #
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(time, emg[0])
    # plt.subplot(212)
    # plt.plot(time, q_knee)
    # plt.subplot(211)
    # plt.plot(time, func(q_knee, q_hip, 1, 1, c))

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_knee, q_hip, emg[i, :], 1, 1)
        print(c)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ax.scatter(q_knee, q_hip, emg[i, :], s=1, color='blue')
        ax.scatter(q_knee, q_hip, func(q_knee, q_hip, 1, 1, c), s=1, color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_knee, q_hip, emg[i, :], 1, 1)
        plt.subplot(2, 3, i + 1)
        plt.plot(q_knee, emg[i, :], color='blue')
        plt.plot(q_knee, func(q_knee, q_hip, 1, 1, c), color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_knee, q_hip, emg[i, :], 1, 1)
        plt.subplot(2, 3, i + 1)
        plt.plot(q_hip, emg[i, :], color='blue')
        plt.plot(q_hip, func(q_knee, q_hip, 1, 1, c), color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_knee, q_hip, emg[i, :], 1, 1)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ang1 = np.linspace(0, 110, 500)
        ang2 = np.linspace(0, 110, 500)
        X, Y = np.meshgrid(ang1, ang2)
        # X = X.flatten()
        # Y = Y.flatten()
        ax.plot_surface(X, Y, new_func(X, Y, i), cmap='rainbow')


def joint_processing_bp_yt(q, time0):
    time = []
    q_1 = []
    q_2 = []
    q_t = np.asarray(q['time'])
    datatime0 = datetime.datetime.strptime(time0, '%H:%M:%S')
    for i in range(len(q_t)):
        datatime = datetime.datetime.strptime(q_t[i], '%H:%M:%S:%f')
        t = (datatime - datatime0).total_seconds()
        if t >= 0:
            time.append(t)
            q_1.append(q['LEF'][i])
            q_2.append(q['LSF'][i])
    time = np.asarray(time)
    q_1 = np.asarray(q_1)
    q_2 = np.asarray(q_2)
    return time, q_1,


def joint_processing_dl_yt(q):
    time = []
    q_1 = []
    q_2 = []
    q_t = np.asarray(q['time'])
    datatime0 = 0
    for i in range(len(q_t)):
        datatime = q_t[i]
        t = datatime - datatime0
        if t >= 0:
            time.append(t)
            q_1.append(q['hip_flexion_r'][i])
            q_2.append(q['knee_angle_r'][i])
    time = np.asarray(time)
    q_1 = np.asarray(q_1)
    q_2 = np.asarray(q_2)
    return time, q_1, q_2


def emg_progressing_bp_yt(emg, fs=1000, people=None, left=True):
    [emg_BIC, t1] = emg_rectification(emg['Biceps'], fs, 'BIC', people, False)
    [emg_TRI, t2] = emg_rectification(emg['Triceps'], fs, 'TRI', people, False)
    [emg_ANT, t3] = emg_rectification(emg['Deltoid'], fs, 'ANT', people, False)
    [emg_POS, t4] = emg_rectification(emg['Medius'], fs, 'POS', people, False)
    [emg_PEC, t5] = emg_rectification(emg['Pectoralis'], fs, 'PEC', people, False)
    [emg_LAT, t6] = emg_rectification(emg['Latissimus'], fs, 'LAT', people, False)
    emg_list = np.asarray([emg_BIC, emg_TRI, emg_ANT, emg_POS, emg_PEC, emg_LAT])
    t_list = np.asarray([t1, t2, t3, t4, t5, t6])
    # return [emg_list, t_list[:, :emg_list.shape[1]]]
    return [emg_list, t1[:emg_list.shape[1]]]


def emg_progressing_dl_yt(emg, fs=1000, people=None, left=True):
    emg = np.asarray(emg)
    emg_rect_label = ['TA', 'GL', 'GM', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'ST', 'BF',
                      'GMax', 'GMed', 'PM', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
    emg_list = []
    for i in range(len(emg_rect_label)):
        [emg_rect, t1] = emg_rectification(emg[:, i + 1], fs, emg_rect_label[i], people, left)
        emg_list.append(emg_rect)
    emg_list = np.asarray(emg_list)
    return [emg_list, t1[:emg_list.shape[1]]]


def time_alignment(q1, e1, time0, Fs_e):
    # q_t, q_1, q_2 = joint_processing_bp_yt(q1, time0)
    # [e1, e_t] = emg_progressing_bp_yt(e1, Fs_e, 'yuetian', False)
    q_t, q_1, q_2 = joint_processing_dl_yt(q1)
    [e1, e_t] = emg_progressing_dl_yt(e1, Fs_e, 'yuetian', False)

    time_end = min(max(q_t), max(e_t))
    if max(q_t) < time_end:
        idx = find_nearest_idx(q_t, time_end)
        q_t = q_t[:idx]
        q_1 = q_1[:idx]
        q_2 = q_2[:idx]
    elif max(e_t) < time_end:
        idx = find_nearest_idx(e_t, time_end)
        e_t = e_t[:idx]
        e1 = e1[:, :idx]

    t = []
    e = [([]) for _ in range(20)]
    for i in range(len(q_t)):
        j = find_nearest_idx(e_t, q_t[i])
        t.append(e_t[j])
        for k in range(20):
            e[k].append(e1[k, j])
    t = np.asarray(t)
    e = np.asarray(e)
    return t, q_1, q_2, e


def bench_press_yuetian():
    Fs_e = 1000
    file_folder = 'files/bench press/yuetian/'
    q1 = pd.read_excel(file_folder + '4_2_18_52_31_log.xlsx')
    q2 = pd.read_excel(file_folder + '4_2_18_52_31_log.xlsx')
    e1 = pd.read_excel(file_folder + 'test 2024_04_02 18_52_55.xlsx')
    e2 = pd.read_excel(file_folder + 'test 2024_04_02 18_52_55.xlsx')
    t1, q1_1, q2_1, e1 = time_alignment(q1, e1, '18:52:55', Fs_e)
    t2, q1_2, q2_2, e2 = time_alignment(q2, e2, '18:52:55', Fs_e)
    t = np.concatenate((t1, t2), axis=0)
    q_1 = np.concatenate((q1_1, q1_2), axis=0)
    q_2 = np.concatenate((q2_1, q2_2), axis=0)
    e = np.concatenate((e1, e2), axis=1)

    plt.figure()
    plt.subplot(211)
    plt.plot(t, q_1)
    plt.plot(t, q_2)
    plt.subplot(212)
    plt.plot(t, e[0])
    plt.plot(t, e[1])

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        print(c)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ax.scatter(q_1, q_2, e[i, :], s=1, color='blue')
        ax.scatter(q_1, q_2, func(q_1, q_2, 1, 1, c), s=1, color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        plt.subplot(2, 3, i + 1)
        plt.plot(q_1, e[i, :], color='blue')
        plt.plot(q_1, func(q_1, q_2, 1, 1, c), color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        plt.subplot(2, 3, i + 1)
        plt.plot(q_2, e[i, :], color='blue')
        plt.plot(q_2, func(q_1, q_2, 1, 1, c), color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ang1 = np.linspace(0, 130, 500)
        ang2 = np.linspace(0, 80, 500)
        X, Y = np.meshgrid(ang1, ang2)
        # X = X.flatten()
        # Y = Y.flatten()
        ax.plot_surface(X, Y, new_func(X, Y, i), cmap='rainbow')


def deadlift_yuetian():
    Fs_e = 1000
    file_folder = 'files/deadlift/yuetian/'
    q1 = pd.read_excel(file_folder + 'Xsens_jointangle_q-65.xlsx')
    q2 = pd.read_excel(file_folder + 'Xsens_jointangle_q-75.xlsx')
    # e1 = pd.read_excel(file_folder + 'test 2024_05_17 16_27_52.xlsx')
    # e2 = pd.read_excel(file_folder + 'test 2024_05_17 16_30_59.xlsx')
    e1 = pd.read_excel(file_folder + 'emg/test 2024_05_17 16_37_38.xlsx')
    e2 = pd.read_excel(file_folder + 'emg/test 2024_05_17 16_41_21.xlsx')
    t1, q1_1, q2_1, e1 = time_alignment(q1, e1, '18:52:55', Fs_e)
    t2, q1_2, q2_2, e2 = time_alignment(q2, e2, '18:52:55', Fs_e)
    t = np.concatenate((t1, t2), axis=0)
    q_1 = np.concatenate((q1_1, q1_2), axis=0)
    q_2 = np.concatenate((q2_1, q2_2), axis=0)
    e = np.concatenate((e1, e2), axis=1)

    plt.figure()
    plt.subplot(211)
    plt.plot(t, q_1)
    plt.plot(t, q_2)
    plt.subplot(212)
    plt.plot(t, e[0])
    plt.plot(t, e[1])

    for i in range(20):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        print(c)

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ax.scatter(q_1, q_2, e[i, :], s=1, color='blue')
        ax.scatter(q_1, q_2, func(q_1, q_2, 1, 1, c), s=1, color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        plt.subplot(2, 3, i + 1)
        plt.plot(q_1, e[i, :], color='blue')
        plt.plot(q_1, func(q_1, q_2, 1, 1, c), color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        plt.subplot(2, 3, i + 1)
        plt.plot(q_2, e[i, :], color='blue')
        plt.plot(q_2, func(q_1, q_2, 1, 1, c), color='red')

    plt.figure()
    for i in range(6):
        c, function = angle_function(q_1, q_2, e[i, :], 1, 1)
        ax = plt.subplot(2, 3, i + 1, projection="3d")
        ang1 = np.linspace(0, 130, 500)
        ang2 = np.linspace(0, 80, 500)
        X, Y = np.meshgrid(ang1, ang2)
        # X = X.flatten()
        # Y = Y.flatten()
        ax.plot_surface(X, Y, new_func(X, Y, i), cmap='rainbow')


if __name__ == '__main__':
    # deadlift_yuetian()

    # plt.show()

    # # x1: shoulder flextion/extension angle
    # # x2: elbow flextion/extension angle
    # # m1: parameters of biceps brachii
    # # m2: parameters of triceps brachii
    # # m3: parameters of anterior deltoid
    # # m4: parameters of medius deltoid
    # # m5: parameters of pectoralis major
    # # m6: parameters of latissimus dorsi
    # m1 = [-2.83847443e-02, 3.78972232e-04, 5.73952879e-04, -4.93123956e-07, -4.39014837e-06, -1.86593838e-06]
    # m2 = [-1.59330326e-01, 2.30448754e-03, 2.35025582e-03, -5.08526370e-06, -2.24674247e-05, -6.13742225e-07]
    # m3 = [-1.37639327e-01, 2.00722246e-03, 2.18952440e-03, -4.10670254e-06, -2.10775861e-05, -2.03264374e-06]
    # m4 = [-2.37691947e-03, 7.78713575e-05, 5.35114834e-05, 5.08632500e-08, -8.37050041e-07, -6.00804626e-08]
    # m5 = [-1.01442551e-01, 1.47043898e-03, 1.56184263e-03, -3.23510388e-06, -1.43861358e-05, -1.25572653e-06]
    # m6 = [-1.07357358e-03, 1.15538266e-04, 2.14702498e-04, -7.47505564e-07, -1.02141327e-07, -2.59922512e-06]

    m = [[2.39143515e-02, -3.04760316e-04, 1.12571905e-03, -7.72324530e-06, 4.09345784e-05, -5.45372943e-05],
         [1.65120025e-02, 3.23591139e-03, 1.59817010e-03, -5.64813163e-05, 1.72556414e-04, -1.99607911e-04],
         [-3.53273111e-03, 1.61506178e-03, 4.19356081e-03, -3.73600143e-05, 1.30932619e-04, -1.98477030e-04],
         [-1.27473005e-02, 2.32431698e-03, 7.52763687e-03, -6.38299366e-05, 1.86197866e-04, -2.68857612e-04],
         [1.05227573e-02, -4.36577687e-05, 1.10565988e-03, -1.03800756e-05, 4.20161018e-05, -4.55842636e-05],
         [-2.26217954e-02, -3.75512215e-04, 5.75974586e-03, -5.38974355e-05, 2.27451771e-04, -2.46057174e-04],
         [-1.80850017e-02, 2.07655842e-03, 2.25215398e-03, -3.03300184e-05, 7.09723443e-05, -1.00875538e-04],
         [-5.79915645e-02, 8.38150778e-04, 6.46074882e-03, -2.32371240e-05, 1.13472730e-04, -1.88573789e-04],
         [-7.21536503e-02, 8.14427689e-03, 3.16312329e-03, -8.09717381e-05, 1.51993754e-04, -2.37976964e-04],
         [-3.16839780e-02, 5.93574143e-03, 2.71470471e-03, -9.87224135e-05, 2.49009181e-04, -2.90078593e-04],
         [1.79046421e-02, 3.04549037e-03, 7.07757681e-03, -8.89327626e-05, 2.69459336e-04, -3.56789595e-04],
         [-5.42379893e-03, 3.22800863e-03, 5.48023549e-03, -7.10426084e-05, 1.95591337e-04, -2.55520098e-04],
         [-2.29667407e-01, 8.86858208e-03, 1.64536883e-02, -1.12149877e-04, 2.20951349e-04, -4.40681813e-04],
         [-2.09985184e-01, 4.09904858e-03, 1.60305670e-02, -9.30262298e-05, 2.99773491e-04, -4.79913194e-04],
         [-1.98342719e-02, -6.99683040e-04, 3.46112995e-03, -3.87014601e-05, 1.98759262e-04, -1.77993923e-04],
         [-2.53127609e-02, 2.03448297e-03, 3.77307591e-03, -5.97754084e-05, 1.91171802e-04, -2.10916063e-04],
         [-4.94494880e-02, 3.28294215e-03, 5.79037580e-03, -8.73302735e-05, 2.62347145e-04, -3.02242308e-04],
         [1.89293401e-02, -3.60671479e-04, -1.61374809e-04, 4.55957310e-06, 1.63040022e-05, -5.31534811e-06],
         [-1.23926566e-02, 4.18773637e-03, 2.43550141e-03, -8.84734560e-05, 2.41207703e-04, -2.34083773e-04],
         [2.54155085e-01, 1.40833780e-02, -2.66909316e-02, -5.88011456e-05, -9.13563325e-05, 3.83919582e-04]]
    m = np.asarray(m)
    # w1 and w2 is related to the people and the motion.
    # w1 * x1 + w2 * x2 is around 100.
    w1 = 1
    w2 = 1
    act = m[i][0] + m[i][1] * x1 * w1 + m[i][2] * x2 * w2 + m[i][3] * x1 ** 2 + m[i][4] * x1 * x2 + m[i][5] * x2 ** 2
