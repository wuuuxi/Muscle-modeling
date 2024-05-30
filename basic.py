import scipy.interpolate as interpolate
import numpy as np
import scipy
import pandas as pd
import os
from require import *

if sport_label == 'biceps_curl':
    if muscle_number == 3:
        if left_or_right == 'left':
            muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l']
        else:
            assert left_or_right == 'right'
            muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1']
        iso = [173, 314, 30]
        OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404]
        MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
        PennationAngleAtOptimal = [0, 0, 0]
        KshapeActive = [0.45, 0.45, 0.45]
    elif muscle_number == 4:
        if left_or_right == 'left':
            muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l', 'tric_long_1_l']
        else:
            assert left_or_right == 'right'
            muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1', 'tric_long_1']
        iso = [173, 314, 30, 223]
        OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404, 0.09027865826431776]
        MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
        PennationAngleAtOptimal = [0, 0, 0, 0.174532925199]
        KshapeActive = [0.45, 0.45, 0.45, 0.45]
    elif muscle_number == 7:
        muscle_idx = ['BICsht', 'BRA', 'BRD', 'BIClon', 'TRIlon', 'TRIlat', 'TRImed']
        iso = [316.8, 1177.37, 276.0, 525.1, 771.8, 717.5, 717.5]
elif sport_label == 'bench_press':
    if muscle_number == 6:
        if left_or_right == 'left':
            muscle_idx = ['bic_s_l', 'tric_long_1_l', 'delt_clav_1_l', 'delt_scap_9_l', 'pect_maj_t_1_l', 'LD_T10_l']
            iso = [173.0, 223.0, 128.0, 286.0, 125.0, 27.6]
        else:
            assert left_or_right == 'right'
            # muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'delt_scap_1', 'pect_maj_t_1', 'LD_T10_r']
            # iso = [173.0, 223.0, 128.0, 289.0, 125.0, 27.6]
            # muscle_idx = ['bic_s', 'tric_long_1', 'delt_scap_1', 'ter_maj_1', 'pect_maj_t_1', 'LD_T10_r']
            # iso = [173.0, 223.0, 128.0, 165.0, 125.0, 27.6]
            muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'ter_maj_1', 'pect_maj_t_1', 'LD_T10_r']
            iso = [173.0, 223.0, 128.0, 165.0, 125.0, 27.6]
        musc_label = ['Biceps', 'Triceps', 'Deltoid', 'Medius', 'Pectoralis', 'Latissimus']
        # iso = [173.0, 223.0, 128.0, 289.0, 125.0, 27.6]
        measured_muscle_idx = muscle_idx
    else:
        if left_or_right == 'left':
            measured_muscle_idx = ['bic_s_l', 'tric_long_1_l', 'delt_clav_1_l', 'delt_scap_9_l', 'pect_maj_t_1_l',
                                   'LD_T10_l']
            muscle_idx = ['bic_s_l', 'tric_long_1_l', 'delt_clav_1_l', 'delt_scap_9_l', 'pect_maj_t_1_l', 'LD_T10_l',
                          'bic_l_l', 'brachialis_1_l', 'brachialis_2_l', 'brachialis_3_l', 'brachialis_4_l',
                          'brachialis_5_l',
                          'brachialis_6_l', 'brachialis_7_l', 'brachiorad_1_l', 'brachiorad_2_l', 'brachiorad_3_l',
                          'tric_long_2_l', 'tric_long_3_l', 'tric_long_4_l',
                          'delt_clav_2_l', 'delt_clav_3_l', 'delt_clav_4_l',
                          'delt_scap_4_l', 'delt_scap_5_l', 'delt_scap_6_l', 'delt_scap_7_l', 'delt_scap_8_l',
                          'delt_scap10_l', 'delt_scap11_l',
                          'pect_maj_t_2_l', 'pect_maj_t_3_l', 'pect_maj_t_4_l', 'pect_maj_t_5_l', 'pect_maj_t_6_l',
                          'LD_T7_l', 'LD_T8_l', 'LD_T9_l', 'LD_T11_l', 'LD_T12_l']
        else:
            measured_muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'delt_scap_9', 'pect_maj_t_1', 'LD_T10_r']
            muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'delt_scap_9', 'pect_maj_t_1', 'LD_T10_r',
                          'bic_l', 'brachialis_1', 'brachialis_2', 'brachialis_3', 'brachialis_4', 'brachialis_5',
                          'brachialis_6', 'brachialis_7', 'brachiorad_1', 'brachiorad_2', 'brachiorad_3',
                          'tric_long_2', 'tric_long_3', 'tric_long_4',
                          'delt_clav_2', 'delt_clav_3', 'delt_clav_4',
                          'delt_scap_4', 'delt_scap_5', 'delt_scap_6', 'delt_scap_7', 'delt_scap_8',
                          'delt_scap10', 'delt_scap11',
                          'pect_maj_t_2', 'pect_maj_t_3', 'pect_maj_t_4', 'pect_maj_t_5', 'pect_maj_t_6',
                          'LD_T7_r', 'LD_T8_r', 'LD_T9_r', 'LD_T11_r', 'LD_T12_r']
        iso = [173.0, 223.0, 128.0, 286.0, 125.0, 27.6,
               173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0,
               223.0, 223.0, 223.0,
               128.0, 128.0, 128.0,
               286.0, 286.0, 286.0, 286.0, 286.0, 286.0, 286.0,
               125.0, 125.0, 125.0, 125.0, 125.0,
               27.6, 27.6, 27.6, 27.6, 27.6]
        musc_label = ['Biceps', 'Triceps', 'Deltoid', 'Medius', 'Pectoralis', 'Latissimus']
elif sport_label == 'deadlift':
    measured_muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
                           'vasmed_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
                           'glmax1_r', 'glmed1_r', 'Ps_L1_VB_r', 'IO1_r', 'rect_abd_r',
                           'IL_L1_r', 'MF_m1s_r', 'EO1_r', 'LTpT_T2_r', 'LD_L2_r']
    muscle_idx = measured_muscle_idx
    # muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
    #               'vasmed_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
    #               'glmax1_r', 'glmed1_r', 'Ps_L1_VB_r', 'IO1_r', 'rect_abd_r',
    #               'IL_L1_r', 'MF_m1s_r', 'EO1_r', 'LTpT_T2_r', 'LD_L2_r',
    #               'bfsh_r',
    #               'glmax2_r', 'glmax3_r',
    #               'glmed2_r', 'glmed3_r',
    #               'Ps_L1_VB_l',
    #               'IO2_r', 'IO3_r', 'IO4_r', 'IO5_r', 'IO6_r', 'IO2_l', 'IO3_l', 'IO4_l', 'IO5_l', 'IO6_r',
    #               'rect_abd_r',
    #               'IL_L2_r', 'IL_L3_r', 'IL_L4_r', 'IL_R5_r', 'IL_R6_r', 'IL_R7_r', 'IL_R8_r', 'IL_R9_r', 'IL_R10_r',
    #               'IL_R11_r', 'IL_R12_r', 'IL_L1_l', 'IL_L2_l', 'IL_L3_l', 'IL_L4_l', 'IL_R5_l', 'IL_R6_l', 'IL_R7_l',
    #               'IL_R8_l', 'IL_R9_l', 'IL_R10_l', 'IL_R11_l', 'IL_R12_l',
    #               'MF_m1t_1_r', 'MF_m1t_2_r', 'MF_m1t_3_r', 'MF_m2s_r', 'MF_m2t_1_r', 'MF_m2t_2_r', 'MF_m2t_3_r',
    #               'MF_m3s_r', 'MF_m3t_1_r', 'MF_m3t_2_r', 'MF_m3t_3_r', 'MF_m4s_r', 'MF_m4t_1_r', 'MF_m4t_2_r',
    #               'MF_m4t_3_r', 'MF_m5s_r', 'MF_m5t_1_r', 'MF_m5t_2_r', 'MF_m5t_3_r', 'MF_m1_laminar_r',
    #               'MF_m2_laminar_r', 'MF_m3_laminar_r', 'MF_m4_laminar_r', 'MF_m5_laminar_r', 'MF_m1s_l', 'MF_m1t_1_l',
    #               'MF_m1t_2_l', 'MF_m1t_3_l', 'MF_m2s_l', 'MF_m2t_1_l', 'MF_m2t_2_l', 'MF_m2t_3_l', 'MF_m3s_l',
    #               'MF_m3t_1_l', 'MF_m3t_2_l', 'MF_m3t_3_l', 'MF_m4s_l', 'MF_m4t_1_l', 'MF_m4t_2_l', 'MF_m4t_3_l',
    #               'MF_m5s_l', 'MF_m5t_1_l', 'MF_m5t_2_l', 'MF_m5t_3_l', 'MF_m1_laminar_l', 'MF_m2_laminar_l',
    #               'MF_m3_laminar_l', 'MF_m4_laminar_l', 'MF_m5_laminar_l',
    #               'EO2_r', 'EO3_r', 'EO4_r', 'EO5_r', 'EO6_r', 'EO1_l', 'EO2_l', 'EO3_l', 'EO4_l', 'EO5_l', 'EO6_l',
    #               'LTpT_T3_r', 'LTpT_T4_r', 'LTpT_T5_r', 'LTpT_T6_r', 'LTpT_T7_r', 'LTpT_T8_r', 'LTpT_T9_r',
    #               'LTpT_T10_r', 'LTpT_T11_r', 'LTpT_T12_r', 'LTpT_R4_r', 'LTpT_R5_r', 'LTpT_R6_r', 'LTpT_R7_r',
    #               'LTpT_R8_r', 'LTpT_R9_r', 'LTpT_R10_r', 'LTpT_R11_r', 'LTpT_R12_r', 'LTpT_T2_l', 'LTpT_T3_l',
    #               'LTpT_T4_l', 'LTpT_T5_l', 'LTpT_T6_l', 'LTpT_T7_l', 'LTpT_T8_l', 'LTpT_T9_l', 'LTpT_T10_l',
    #               'LTpT_T11_l', 'LTpT_T12_l', 'LTpT_R4_l', 'LTpT_R5_l', 'LTpT_R6_l', 'LTpT_R7_l', 'LTpT_R8_l',
    #               'LTpT_R9_l', 'LTpT_R10_l', 'LTpT_R11_l', 'LTpT_R12_l',
    #               'LD_L3_r', 'LD_L4_r', 'LD_L5_r', 'LD_Il_r', 'LD_L2_l', 'LD_L3_l', 'LD_L4_l', 'LD_L5_l', 'LD_Il_l']
    # muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
    #               'vasmed_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
    #               'glmax1_r', 'glmed1_r', 'Ps_L1_VB_r', 'IO1_r', 'rect_abd_r',
    #               'IL_L1_r', 'MF_m1s_r', 'EO1_r', 'LTpT_T2_r', 'LD_L2_r',
    #               'bfsh_r',
    #               'glmax2_r', 'glmax3_r',
    #               'glmed2_r', 'glmed3_r',
    #               'Ps_L1_TP_r', 'Ps_L1_L2_IVD_r', 'Ps_L2_TP_r', 'Ps_L2_L3_IVD_r', 'Ps_L3_TP_r',
    #               'Ps_L3_L4_IVD_r', 'Ps_L4_TP_r', 'Ps_L4_L5_IVD_r', 'Ps_L5_TP_r', 'Ps_L5_VB_r',
    #               'Ps_L1_VB_l', 'Ps_L1_TP_l', 'Ps_L1_L2_IVD_l', 'Ps_L2_TP_l', 'Ps_L2_L3_IVD_l',
    #               'Ps_L3_TP_l', 'Ps_L3_L4_IVD_l', 'Ps_L4_TP_l', 'Ps_L4_L5_IVD_l', 'Ps_L5_TP_l', 'Ps_L5_VB_l',
    #               'IO2_r', 'IO3_r', 'IO4_r', 'IO5_r', 'IO6_r',
    #               'rect_abd_r',
    #               'IL_L2_r', 'IL_L3_r', 'IL_L4_r', 'IL_R5_r', 'IL_R6_r', 'IL_R7_r', 'IL_R8_r', 'IL_R9_r', 'IL_R10_r',
    #               'IL_R11_r', 'IL_R12_r', 'IL_L1_l', 'IL_L2_l', 'IL_L3_l', 'IL_L4_l', 'IL_R5_l', 'IL_R6_l', 'IL_R7_l',
    #               'IL_R8_l', 'IL_R9_l', 'IL_R10_l', 'IL_R11_l', 'IL_R12_l',
    #               'MF_m1t_1_r', 'MF_m1t_2_r', 'MF_m1t_3_r', 'MF_m2s_r', 'MF_m2t_1_r', 'MF_m2t_2_r', 'MF_m2t_3_r',
    #               'MF_m3s_r', 'MF_m3t_1_r', 'MF_m3t_2_r', 'MF_m3t_3_r', 'MF_m4s_r', 'MF_m4t_1_r', 'MF_m4t_2_r',
    #               'MF_m4t_3_r', 'MF_m5s_r', 'MF_m5t_1_r', 'MF_m5t_2_r', 'MF_m5t_3_r', 'MF_m1_laminar_r',
    #               'MF_m2_laminar_r', 'MF_m3_laminar_r', 'MF_m4_laminar_r', 'MF_m5_laminar_r', 'MF_m1s_l', 'MF_m1t_1_l',
    #               'MF_m1t_2_l', 'MF_m1t_3_l', 'MF_m2s_l', 'MF_m2t_1_l', 'MF_m2t_2_l', 'MF_m2t_3_l', 'MF_m3s_l',
    #               'MF_m3t_1_l', 'MF_m3t_2_l', 'MF_m3t_3_l', 'MF_m4s_l', 'MF_m4t_1_l', 'MF_m4t_2_l', 'MF_m4t_3_l',
    #               'MF_m5s_l', 'MF_m5t_1_l', 'MF_m5t_2_l', 'MF_m5t_3_l', 'MF_m1_laminar_l', 'MF_m2_laminar_l',
    #               'MF_m3_laminar_l', 'MF_m4_laminar_l', 'MF_m5_laminar_l',
    #               'EO2_r', 'EO3_r', 'EO4_r', 'EO5_r', 'EO6_r', 'EO1_l', 'EO2_l', 'EO3_l', 'EO4_l', 'EO5_l', 'EO6_l',
    #               'LTpT_T3_r', 'LTpT_T4_r', 'LTpT_T5_r', 'LTpT_T6_r', 'LTpT_T7_r', 'LTpT_T8_r', 'LTpT_T9_r',
    #               'LTpT_T10_r', 'LTpT_T11_r', 'LTpT_T12_r', 'LTpT_R4_r', 'LTpT_R5_r', 'LTpT_R6_r', 'LTpT_R7_r',
    #               'LTpT_R8_r', 'LTpT_R9_r', 'LTpT_R10_r', 'LTpT_R11_r', 'LTpT_R12_r', 'LTpT_T2_l', 'LTpT_T3_l',
    #               'LTpT_T4_l', 'LTpT_T5_l', 'LTpT_T6_l', 'LTpT_T7_l', 'LTpT_T8_l', 'LTpT_T9_l', 'LTpT_T10_l',
    #               'LTpT_T11_l', 'LTpT_T12_l', 'LTpT_R4_l', 'LTpT_R5_l', 'LTpT_R6_l', 'LTpT_R7_l', 'LTpT_R8_l',
    #               'LTpT_R9_l', 'LTpT_R10_l', 'LTpT_R11_l', 'LTpT_R12_l',
    #               'LD_L3_r', 'LD_L4_r', 'LD_L5_r', 'LD_Il_r', 'LD_L2_l', 'LD_L3_l', 'LD_L4_l', 'LD_L5_l', 'LD_Il_l']
    iso = [5] * 183
    musc_label = ['TA', 'GL', 'GM', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'ST', 'BF',
                  'GMax', 'GMed', 'PM', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']


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


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx


def emg_rectification(x, Fs=1000, code=None, people=None, left=False):
    # Fs 采样频率，在港大的EMG信号中是1000Hz
    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 5  # 低通滤波4—10Hz得到包络线

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
    if people == 'chenzui':  # left
        if code == 'BIC':
            ref = 407.11
        elif code == 'BRA':
            ref = 250.37
        elif code == 'BRD':
            ref = 468.31
        elif code == 'TRI':
            ref = 467.59
        elif code == 'ANT':
            ref = 176.04
        elif code == 'POS':
            ref = 272.30
        elif code == 'PEC':
            ref = 87.11
        elif code == 'LAT':
            ref = 339.21
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
        elif code == 'ANT':
            ref = 280.711
        elif code == 'POS':
            ref = 116.263
        elif code == 'PEC':
            ref = 223.172
        elif code == 'LAT':
            ref = 108.985 * 10
        else:
            ref = max(EMGLE)
    elif people == 'yuetian':  # left
        if left is True:
            if code == 'LTA':
                ref = 3.989497
            elif code == 'LGL':
                ref = 3.319789
            elif code == 'LVL':
                ref = 4.036685
            elif code == 'LRF':
                ref = 3.125095
            elif code == 'LST':
                ref = 3.45323
            elif code == 'LBF':
                ref = 4.025123
            else:
                ref = max(EMGLE)
        else:
            deadlift_musc_label = ['TA', 'GL', 'GM', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'ST', 'BF',
                                   'GMax', 'GMed', 'PM', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
            deadlift_mvc = [0.1747, 0.1221, 0.1204, 0.1607, 0.1421, 0.1202, 0.1728, 0.1037, 0.2073, 0.1656,
                            0.0406, 0.0511, 0.0735, 0.1457, 0.1567, 0.1388, 0.1811, 0.1679, 0.1136, 0.0754]
            # deadlift_mvc = [0.1747, 0.1221, 0.1204, 0.1607, 0.1421, 0.1202, 0.0728, 0.1037, 0.2073, 0.1656,
            #                 0.0406, 0.0511, 0.0735, 0.1457, 0.1167, 0.1388, 0.0811, 0.0679, 0.1136, 0.0754]
            if code == 'BIC':
                ref = 0.67077
            elif code == 'TRI':
                ref = 0.67077
                # ref = 0.33346
            elif code == 'ANT':
                ref = 0.263597
            elif code == 'POS':
                ref = 0.23831
            elif code == 'PEC':
                ref = 0.43018
            elif code == 'LAT':
                ref = 0.15176
            elif code == 'BRA':
                ref = 0.67077
            elif code == 'BRD':
                ref = 0.67077
            elif code in deadlift_musc_label:
                index = deadlift_musc_label.index(code)
                ref = deadlift_mvc[index]
            else:
                ref = 2
            # ref = 2
    else:
        print('No information of this people.')
    # ref = 1

    normalized_EMG = EMGLE / ref
    y = normalized_EMG
    return [y, t]


def elbow_emg(emg1, emg2):
    # emg1: Biceps
    # emg2: Triceps
    x = np.asarray([[5.79835623, 1.40694018, 0.26443844, 1.34689293],
                    [0.17944381, 1.81618533, 4.21557155, 0.0]])
    # x = np.asarray([[5.02488458, 0.67736425, 1.00691882, 0.47668198],
    #                 [0.0, 2.30828341, 0.47426132, 2.30039555]])
    emg1 = np.asarray(emg1)
    emg2 = np.asarray(emg2)
    s1 = -(x[1, 2] * emg2 - x[1, 3] * emg1) / (x[0, 2] * x[1, 3] - x[1, 2] * x[0, 3])
    s2 = (x[0, 2] * emg2 - x[0, 3] * emg1) / (x[0, 2] * x[1, 3] - x[1, 2] * x[0, 3])
    a1 = s1 * x[0, 0] + s2 * x[1, 0]
    a2 = s1 * x[0, 1] + s2 * x[1, 1]
    return a1, a2


def from_csv_to_xlsx(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if 'csv' in file:
                xlsx_file = file[:-4] + '.xlsx'
                # 由于处理HKU的emg，从第6行开始读
                # data_frame = pd.read_csv(root + '/' + file, skiprows=5)
                data_frame = pd.read_csv(root + '/' + file)
                data_frame.to_excel(root + '/' + xlsx_file, index=False)


if __name__ == '__main__':
    # from_csv_to_xlsx('files/bench press/kehan/Kehan bench')
    from_csv_to_xlsx('files/bench press/yuetian/0408/20240408健身机数据')
