import os

import numpy as np
import pandas as pd

from require import *
from basic import *
import matplotlib.pyplot as plt


def read_realted_files(label='chenzui-left-3kg', idx='1', include_state=include_state):
    '''
    :param label:
    :param idx:
    :return: emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    '''
    fs = 2000
    ID = 'inverse_dynamics-'
    JA = 'Xsens_jointangle_q-'
    LE = 'modelModified_MuscleAnalysis_Length-'
    TL = 'modelModified_MuscleAnalysis_TendonLength-'
    if sport_label == 'biceps_curl':
        MA = 'modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-'
    elif sport_label == 'bench_press':
        if left_or_right == 'left':
            MA = 'Subject-test-scaled_MuscleAnalysis_MomentArm_arm_flex_l-'
        else:
            MA = 'Subject-test-scaled_MuscleAnalysis_MomentArm_arm_flex_r-'

    if label == 'chenzui-left-3kg':
        people = 'chenzui'
        assert idx in ['11', '13', '14', '15', '17', '18', '19']
        file_folder = 'files/chenzui-3kg-1/'
        emg = np.load(file_folder + 'emg-' + idx + '.npy')
        moment = pd.read_excel(file_folder + ID + idx + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + idx + '.xlsx')
        # length = pd.read_excel(file_folder + LE + idx + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + idx + '.xlsx')
    elif label == 'chenzui-left-5.5kg':
        people = 'chenzui'
        file_folder = 'files/chenzui-5.5kg-1/'
        if idx == '2':
            emg = np.load(file_folder + '2.npy')
            moment = pd.read_excel(file_folder + ID + '2.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-2.xlsx')
        elif idx == '3':
            emg = np.load(file_folder + '3.npy')
            moment = pd.read_excel(file_folder + ID + '3.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-3.xlsx')
        elif idx == '4':
            emg = np.load(file_folder + '4.npy')
            moment = pd.read_excel(file_folder + ID + '4.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-4.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-4.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-4.xlsx')
        elif idx == '5':
            emg = np.load(file_folder + '5.npy')
            moment = pd.read_excel(file_folder + ID + '5.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-5.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-5.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-5.xlsx')
        elif idx == '6':
            emg = np.load(file_folder + '6.npy')
            moment = pd.read_excel(file_folder + ID + '6.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-6.xlsx')
        elif idx == '7':
            emg = np.load(file_folder + '7.npy')
            moment = pd.read_excel(file_folder + ID + '7.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-7.xlsx')
        else:
            print('No such index!')
    elif label == 'zhuo-left-1kg':
        people = 'zhuo'
        file_folder = 'files/lizhuo-1kg/'
        if idx == '2':
            emg = np.load(file_folder + '2.npy')
            moment = pd.read_excel(file_folder + ID + '2.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-2.xlsx')
        elif idx == '3':
            emg = np.load(file_folder + '3.npy')
            moment = pd.read_excel(file_folder + ID + '3.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-3.xlsx')
        elif idx == '4':
            emg = np.load(file_folder + '4.npy')
            moment = pd.read_excel(file_folder + ID + '4.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-4.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-4.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-4.xlsx')
        elif idx == '6':
            emg = np.load(file_folder + '6.npy')
            moment = pd.read_excel(file_folder + ID + '6.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-6.xlsx')
        elif idx == '7':
            emg = np.load(file_folder + '7.npy')
            moment = pd.read_excel(file_folder + ID + '7.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-7.xlsx')
        elif idx == '9':
            emg = np.load(file_folder + '9.npy')
            moment = pd.read_excel(file_folder + ID + '9.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-9.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-9.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-9.xlsx')
        else:
            print('No such index!')
    elif label == 'zhuo-right-3kg':
        people = 'zhuo'
        file_folder = 'files/lizhuo-3kg/'
        if idx == '1':
            emg = np.load(file_folder + '1.npy')
            moment = pd.read_excel(file_folder + ID + '1.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-1.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-1.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-1.xlsx')
        elif idx == '2':
            emg = np.load(file_folder + '2.npy')
            moment = pd.read_excel(file_folder + ID + '2.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-2.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-2.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-2.xlsx')
        elif idx == '3':
            emg = np.load(file_folder + '3.npy')
            moment = pd.read_excel(file_folder + ID + '3.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-3.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-3.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-3.xlsx')
        elif idx == '5':
            emg = np.load(file_folder + '5.npy')
            moment = pd.read_excel(file_folder + ID + '5.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-5.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-5.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-5.xlsx')
        elif idx == '6':
            emg = np.load(file_folder + '6.npy')
            moment = pd.read_excel(file_folder + ID + '6.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-6.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-6.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-6.xlsx')
        elif idx == '7':
            emg = np.load(file_folder + '7.npy')
            moment = pd.read_excel(file_folder + ID + '7.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-7.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-7.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-7.xlsx')
        elif idx == '8':
            emg = np.load(file_folder + '8.npy')
            moment = pd.read_excel(file_folder + ID + '8.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-8.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-8.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-8.xlsx')
        elif idx == '9':
            emg = np.load(file_folder + '9.npy')
            moment = pd.read_excel(file_folder + ID + '9.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-9.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-9.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-9.xlsx')
        elif idx == '10':
            emg = np.load(file_folder + '10.npy')
            moment = pd.read_excel(file_folder + ID + '10.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-10.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-10.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-10.xlsx')
        elif idx == '11':
            emg = np.load(file_folder + '10.npy')
            moment = pd.read_excel(file_folder + ID + '11.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-11.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-11.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-11.xlsx')
        elif idx == '12-1' or idx == '12-2' or idx == '12-3':
            emg = np.load(file_folder + '12.npy')
            moment = pd.read_excel(file_folder + ID + '12.xlsx')
            length = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_Length-12.xlsx')
            tenlen = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_TendonLength-12.xlsx')
            momarm = pd.read_excel(file_folder + 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_r-12.xlsx')
    elif label == 'chenzui-left-all-2.5kg':
        people = 'chenzui'
        assert idx in ['5', '6', '7', '8', '9']
        file_folder = 'files/chenzui-all/'
        emg = np.load(file_folder + '2.5-' + idx + '.npy')
        moment = pd.read_excel(file_folder + ID + idx + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + idx + '.xlsx')
        # length = pd.read_excel(file_folder + LE + idx + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + idx + '.xlsx')
    elif label == 'chenzui-left-all-3kg':
        people = 'chenzui'
        assert idx in ['10', '11', '12', '13', '14']
        file_folder = 'files/chenzui-all/'
        emg = np.load(file_folder + '3-' + idx + '.npy')
        moment = pd.read_excel(file_folder + ID + idx + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + idx + '.xlsx')
        # length = pd.read_excel(file_folder + LE + idx + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + idx + '.xlsx')
    elif label == 'chenzui-left-all-4kg':
        people = 'chenzui'
        assert idx in ['15', '16', '17']
        file_folder = 'files/chenzui-all/'
        emg = np.load(file_folder + '4-' + idx + '.npy')
        moment = pd.read_excel(file_folder + ID + idx + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + idx + '.xlsx')
        # length = pd.read_excel(file_folder + LE + idx + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + idx + '.xlsx')
    elif label == 'chenzui-left-all-6.5kg':
        people = 'chenzui'
        assert idx in ['18', '19', '20', '21', '22']
        file_folder = 'files/chenzui-all/'
        emg = np.load(file_folder + '6.5-' + idx + '.npy')
        moment = pd.read_excel(file_folder + ID + idx + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + idx + '.xlsx')
        # length = pd.read_excel(file_folder + LE + idx + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + idx + '.xlsx')
        emg_mean = np.load('emg/chenzui_mean_6.5kg.npy')
        emg_std = np.load('emg/chenzui_std_6.5kg.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_6.5kg.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_6.5kg.npy')
    # elif label == 'chenzui-left-all-6.5kg-cts':
    #     people = 'chenzui'
    #     assert idx in ['23-1', '23-2', '23-3', '23-4', '23-5']
    #     file_folder = 'files/chenzui-all/'
    #     emg = np.load(file_folder + '6.5' + '.npy')
    #     moment = pd.read_excel(file_folder + ID + '23' + '.xlsx')
    #     momarm = pd.read_excel(file_folder + MA + '23' + '.xlsx')
    #     # length = pd.read_excel(file_folder + LE + '23' + '.xlsx')
    #     # tenlen = pd.read_excel(file_folder + TL + '23' + '.xlsx')
    #     emg_mean = np.load('emg/chenzui_mean_6.5kg_cts.npy')
    #     emg_std = np.load('emg/chenzui_std_6.5kg_cts.npy')
    #     emg_trend_u = np.load('emg/chenzui_trend_u_6.5kg_cts.npy')
    #     emg_trend_d = np.load('emg/chenzui_trend_d_6.5kg_cts.npy')
    elif label == 'chenzui-left-all-4kg-cts':
        people = 'chenzui'
        assert idx in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        file_folder = 'files/chenzui-cts/'
        emg = np.load(file_folder + '5-4' + '.npy')
        moment = pd.read_excel(file_folder + ID + '5' + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + '5' + '.xlsx')
        # length = pd.read_excel(file_folder + LE + '5' + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + '5' + '.xlsx')
        emg_mean = np.load('emg/chenzui_mean_4kg_cts.npy')
        emg_std = np.load('emg/chenzui_std_4kg_cts.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_4kg_cts.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_4kg_cts.npy')
    elif label == 'chenzui-left-all-5.5kg-cts':
        people = 'chenzui'
        assert idx in ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        file_folder = 'files/chenzui-cts/'
        emg = np.load(file_folder + '4-5.5' + '.npy')
        moment = pd.read_excel(file_folder + ID + '4' + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + '4' + '.xlsx')
        # length = pd.read_excel(file_folder + LE + '5' + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + '5' + '.xlsx')
        emg_mean = np.load('emg/chenzui_mean_5.5kg_cts.npy')
        emg_std = np.load('emg/chenzui_std_5.5kg_cts.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_5.5kg_cts.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_5.5kg_cts.npy')
    elif label == 'chenzui-left-all-6.5kg-cts':
        people = 'chenzui'
        assert idx in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        file_folder = 'files/chenzui-cts/'
        emg = np.load(file_folder + '2-6.5' + '.npy')
        moment = pd.read_excel(file_folder + ID + '2' + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + '2' + '.xlsx')
        # length = pd.read_excel(file_folder + LE + '5' + '.xlsx')
        # tenlen = pd.read_excel(file_folder + TL + '5' + '.xlsx')
        emg_mean = np.load('emg/chenzui_mean_6.5kg_cts.npy')
        emg_std = np.load('emg/chenzui_std_6.5kg_cts.npy')
        emg_trend_u = np.load('emg/chenzui_trend_u_6.5kg_cts.npy')
        emg_trend_d = np.load('emg/chenzui_trend_d_6.5kg_cts.npy')
    elif label == 'bp-chenzui-left-4kg':
        people = 'chenzui'
        assert idx in ['1', '2', '3', '4', '5', '6']
        file_folder = 'files/bench press/chenzui/'
        emg = np.load(file_folder + '4' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '4' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '4' + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + '4' + '.xlsx')
        emg_mean = np.load('emg/bp-chenzui_mean_4kg.npy')
        emg_std = np.load('emg/bp-chenzui_std_4kg.npy')
        emg_trend_u = np.load('emg/bp-chenzui_trend_u_4kg.npy')
        emg_trend_d = np.load('emg/bp-chenzui_trend_d_4kg.npy')
    elif label == 'bp-zhuo-right-3kg':
        people = 'zhuo'
        assert idx in ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        file_folder = 'files/bench press/lizhuo/'
        emg = np.load(file_folder + '3' + '.npy')
        moment = pd.read_excel(file_folder + ID + '3' + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + '3' + '.xlsx')
        emg_mean = np.load('emg/bp-lizhuo_mean_3kg.npy')
        emg_std = np.load('emg/bp-lizhuo_std_3kg.npy')
        emg_trend_u = np.load('emg/bp-lizhuo_trend_u_3kg.npy')
        emg_trend_d = np.load('emg/bp-lizhuo_trend_d_3kg.npy')
    elif label == 'bp-zhuo-right-4kg':
        people = 'zhuo'
        assert idx in ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        file_folder = 'files/bench press/lizhuo/'
        emg = np.load(file_folder + '4' + '.npy')
        emg[:, 6] = emg[:, 5]
        moment = pd.read_excel(file_folder + ID + '4' + '.xlsx')
        momarm = pd.read_excel(file_folder + MA + '4' + '.xlsx')
        emg_mean = np.load('emg/bp-lizhuo_mean_4kg.npy')
        emg_std = np.load('emg/bp-lizhuo_std_4kg.npy')
        emg_trend_u = np.load('emg/bp-lizhuo_trend_u_4kg.npy')
        emg_trend_d = np.load('emg/bp-lizhuo_trend_d_4kg.npy')
    else:
        print('No corresponding label!')
        return 0

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
    # elif label == 'chenzui-left-all-2.5kg' or label == 'chenzui-left-all-3kg' or label == 'chenzui-left-all-4kg' or label == 'chenzui-left-all-6.5kg':
    #     emg_mean = np.load('emg/chenzui_mean_all.npy')
    #     emg_std = np.load('emg/chenzui_std_all.npy')
    #     emg_trend_u = np.load('emg/chenzui_trend_u_all.npy')
    #     emg_trend_d = np.load('emg/chenzui_trend_d_all.npy')

    time_torque = moment['time']
    time_momarm = momarm['time']
    time_angle = joint_angle['time']
    angle = joint_angle['arm_flex_l']
    if sport_label == 'biceps_curl':
        if left_or_right == 'right':
            torque = moment['elbow_flex_r_moment']
        else:
            torque = moment['elbow_flex_l_moment']
    elif sport_label == 'bench_press':
        if left_or_right == 'right':
            torque = moment['arm_flex_r_moment']
        else:
            torque = moment['arm_flex_l_moment']

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
    # elif label == 'chenzui-left-all-6.5kg-cts':
    #     if idx == '23-1':
    #         timestep_emg = [18.016, 28.232, 28.232, 38.849]
    #     elif idx == '23-2':
    #         timestep_emg = [47.148, 59.531, 59.531, 70.264]
    #     elif idx == '23-3':
    #         timestep_emg = [78.747, 92.862, 92.862, 104.112]
    #     elif idx == '23-4':
    #         timestep_emg = [112.511, 123.794, 123.794, 133.594]
    #     elif idx == '23-5':
    #         timestep_emg = [141.76, 153.243, 153.243, 162.176]
    elif label == 'chenzui-left-all-4kg-cts':
        if idx == '1':
            timestep_emg = [94.278, 101.394, 102.828, 110.361]
        elif idx == '2':
            timestep_emg = [111.81, 119.16, 120.043, 128.026]
        elif idx == '3':
            timestep_emg = [129.809, 136.259, 137.209, 144.775]
        elif idx == '4':
            timestep_emg = [146.592, 153.675, 154.458, 161.574]
        elif idx == '5':
            timestep_emg = [163.774, 170.157, 171.207, 178.307]
        elif idx == '6':
            timestep_emg = [181.373, 187.506, 188.44, 194.589]
        elif idx == '7':
            timestep_emg = [197.923, 203.989, 204.689, 210.789]
        elif idx == '8':
            timestep_emg = [213.938, 219.421, 220.321, 225.204]
        elif idx == '9':
            timestep_emg = [229.888, 235.954, 236.854, 243.054]
        elif idx == '10':
            timestep_emg = [246.603, 252.453, 253.42, 259.369]
        elif idx == '11':
            timestep_emg = [263.802, 270.019, 270.835, 276.568]
        elif idx == '12':
            timestep_emg = [280.901, 286.851, 287.751, 293.584]
        elif idx == '13':
            timestep_emg = [298.084, 304.317, 305.25, 310.983]
        elif idx == '14':
            timestep_emg = [315.883, 322.199, 322.999, 328.382]
        elif idx == '15':
            timestep_emg = [331.932, 337.632, 338.582, 343.548]
        elif idx == '16':
            timestep_emg = [348.481, 353.631, 354.631, 359.481]
    elif label == 'chenzui-left-all-5.5kg-cts':
        if idx == '1':
            timestep_emg = [15.015, 23.482, 23.881, 32.714]
        elif idx == '2':
            timestep_emg = [34.464, 44.23, 44.464, 53.663]
        elif idx == '3':
            timestep_emg = [56.463, 65.763, 66.696, 75.962]
        elif idx == '4':
            timestep_emg = [78.612, 86.495, 86.962, 94.645]
        elif idx == '5':
            timestep_emg = [99.078, 106.594, 107.344, 114.027]
        elif idx == '6':
            timestep_emg = [118.827, 125.426, 125.76, 131.693]
        elif idx == '7':
            timestep_emg = [137.409, 144.575, 144.642, 150.908]
        elif idx == '8':
            timestep_emg = [155.875, 161.241, 161.491, 166.324]
        elif idx == '9':
            timestep_emg = [171.507, 175.94, 176.357, 181.09]
    elif label == 'chenzui-left-all-6.5kg-cts':
        if idx == '1':
            timestep_emg = [14.966, 21.699, 25.899, 35.565]
        elif idx == '2':
            timestep_emg = [36.365, 43.598, 47.098, 56.881]
        elif idx == '3':
            timestep_emg = [57.98, 64.464, 68.996, 77.879]
        elif idx == '4':
            timestep_emg = [79.33, 85.529, 89.429, 97.562]
        elif idx == '5':
            timestep_emg = [99.412, 106.145, 109.745, 119.378]
        elif idx == '6':
            timestep_emg = [122.227, 127.644, 132.444, 139.76]
        elif idx == '7':
            timestep_emg = [143.56, 148.509, 152.726, 159.975]
        elif idx == '8':
            timestep_emg = [164.809, 171.175, 173.458, 180.291]
        elif idx == '9':
            timestep_emg = [184.341, 190.107, 191.307, 197.44]
        elif idx == '10':
            timestep_emg = [201.807, 206.856, 207.889, 213.623]
    elif label == 'bp-chenzui-left-4kg':
        if idx == '1':
            timestep_emg = [21.999, 26.499, 26.499, 27.898]
        elif idx == '2':
            timestep_emg = [28.632, 32.732, 32.732, 34.031]
        elif idx == '3':
            timestep_emg = [34.498, 38.031, 38.031, 39.365]
        elif idx == '4':
            timestep_emg = [46.797, 50.997, 50.997, 53.364]
        elif idx == '5':
            timestep_emg = [54.164, 57.564, 57.564, 60.33]
        elif idx == '6':
            timestep_emg = [60.33, 64.363, 64.363, 67.296]
    elif label == 'bp-zhuo-right-3kg':
        if idx == '1':
            timestep_emg = [34.131, 36.231, 36.231, 41.498]
        elif idx == '2':
            timestep_emg = [43.364, 44.364, 44.364, 47.197]
        elif idx == '3':
            timestep_emg = [56.997, 58.297, 58.297, 62.063]
        elif idx == '4':
            timestep_emg = [63.330, 64.430, 64.430, 69.663]
        elif idx == '5':
            timestep_emg = [70.963, 72.763, 73.429, 77.162]
        elif idx == '6':
            timestep_emg = [78.629, 80.062, 80.062, 84.329]
        elif idx == '7':
            timestep_emg = [85.929, 87.495, 87.495, 92.628]
        elif idx == '8':
            timestep_emg = [104.361, 106.627, 106.627, 111.46]
        elif idx == '9':
            timestep_emg = [114.094, 116.527, 116.893, 120.86]
    elif label == 'bp-zhuo-right-4kg':
        if idx == '1':
            timestep_emg = [26.132, 28.999, 28.999, 33.232]
        elif idx == '2':
            timestep_emg = [35.032, 37.165, 37.165, 41.765]
        elif idx == '3':
            timestep_emg = [43.631, 46.098, 46.098, 49.365]
        elif idx == '4':
            timestep_emg = [51.698, 53.731, 53.731, 57.464]
        elif idx == '5':
            timestep_emg = [59.697, 61.264, 61.264, 63.93]
        elif idx == '6':
            timestep_emg = [66.497, 68.063, 68.063, 70.93]
        elif idx == '7':
            timestep_emg = [73.263, 74.663, 74.663, 77.23]
        elif idx == '8':
            timestep_emg = [81.03, 82.263, 82.263, 85.096]
        elif idx == '9':
            timestep_emg = [87.996, 89.696, 89.696, 92.929]

    t_tor = []
    t_arm = []
    t_ang = []
    tor = []
    ang = []
    arm = [[] for _ in range(len(muscle_idx))]
    ml = [[] for _ in range(len(muscle_idx))]
    tl = [[] for _ in range(len(muscle_idx))]
    fa = [[] for _ in range(len(muscle_idx))]
    fr = []

    for i in range(int(len(timestep_emg) / 2)):
        tas = find_nearest_idx(time_angle, timestep_emg[2 * i])
        tae = find_nearest_idx(time_angle, timestep_emg[2 * i + 1])
        t_ang.append(resample_by_len(list(time_angle[tas:tae]), target_len))
        ang.append(resample_by_len(list(angle[tas:tae]), target_len))

        tts = find_nearest_idx(time_torque, timestep_emg[2 * i])
        tte = find_nearest_idx(time_torque, timestep_emg[2 * i + 1])
        t_tor.append(resample_by_len(list(time_torque[tts:tte]), target_len))
        tor.append(resample_by_len(list(torque[tts:tte]), target_len))

        tms = find_nearest_idx(time_momarm, timestep_emg[2 * i])
        tme = find_nearest_idx(time_momarm, timestep_emg[2 * i + 1])
        t_arm.append(resample_by_len(list(time_momarm[tms:tme]), target_len))
        for j in range(len(muscle_idx)):
            if arm_constant is True:
                marm = np.ones_like(momarm[muscle_idx[j]][tms:tme]) * np.mean(momarm[muscle_idx[j]][tms:tme])
                arm[j].append(resample_by_len(list(marm), target_len))
            else:
                arm[j].append(resample_by_len(list(momarm[muscle_idx[j]][tms:tme]), target_len))
            # ml[j].append(resample_by_len(list(length[muscle_idx[j]][tms:tme]), target_len))
            # tl[j].append(resample_by_len(list(tenlen[muscle_idx[j]][tms:tme]), target_len))
            # for k in range(tms, tme):
            #     fr.append(calc_fal(length[muscle_idx[j]][k], tenlen[muscle_idx[j]][k], muscle_idx[j]))
            # fa[j].append(resample_by_len(fr, target_len))
            fr = []

    t_tor_out = []
    t_arm_out = []
    t_ang_out = []
    emg_mean_out = []
    emg_std_out = []
    emg_trend_u_out = []
    emg_trend_d_out = []
    tor_out = []
    ang_out = []
    arm_out = [[] for _ in range(len(muscle_idx))]
    ml_out = [[] for _ in range(len(muscle_idx))]
    tl_out = [[] for _ in range(len(muscle_idx))]
    fa_out = [[] for _ in range(len(muscle_idx))]
    for i in range(len(muscle_idx)):
        if include_state == 'lift and down':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :], target_len * 2))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :], target_len * 2))
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len * 2))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len * 2))
            arm_out[i].append(np.concatenate([arm[i][0], arm[i][1]]))
            # ml_out[i].append(np.concatenate([ml[i][0], ml[i][1]]))
            # tl_out[i].append(np.concatenate([tl[i][0], tl[i][1]]))
            # fa_out[i].append(np.concatenate([fa[i][0], fa[i][1]]))
        elif include_state == 'lift':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :emg_lift_len], target_len))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :emg_lift_len], target_len))
            # emg_mean = emg_mean[:, :emg_lift_len]
            # emg_std = emg_std[:, :emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :emg_lift_len], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :emg_lift_len], target_len))
            arm_out[i].append(arm[i][0])
            # ml_out[i].append(ml[i][0])
            # tl_out[i].append(tl[i][0])
            # fa_out[i].append(fa[i][0])
        elif include_state == 'down':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, 0:emg_lift_len], target_len))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, 0:emg_lift_len], target_len))
            emg_mean = emg_mean[:, 0:emg_lift_len]
            emg_std = emg_std[:, 0:emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len))
            arm_out[i].append(arm[i][1])
            # ml_out[i].append(ml[i][1])
            # tl_out[i].append(tl[i][1])
            # fa_out[i].append(fa[i][1])
        else:
            return os.error('State must include lift or down!')

    if include_state == 'lift and down':
        t_tor_out.append(np.concatenate([t_tor[0], t_tor[1]]))
        t_arm_out.append(np.concatenate([t_arm[0], t_arm[1]]))
        t_ang_out.append(np.concatenate([t_ang[0], t_ang[1]]))
        tor_out.append(np.concatenate([tor[0], tor[1]]))
        ang_out.append(np.concatenate([ang[0], ang[1]]))
    elif include_state == 'lift':
        t_tor_out.append(t_tor[0])
        t_arm_out.append(t_arm[0])
        t_ang_out.append(t_ang[0])
        tor_out.append(tor[0])
        ang_out.append(ang[0])
    elif include_state == 'down':
        t_tor_out.append(t_tor[1])
        t_arm_out.append(t_arm[1])
        t_ang_out.append(t_ang[1])
        tor_out.append(tor[1])
        ang_out.append(ang[1])

    if sport_label == 'biceps_curl':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people)
        [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', people)
        [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', people)
        if include_TRI is True:
            [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', people)
    elif sport_label == 'bench_press':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people)
        [emg_TRI, t2] = emg_rectification(emg[:, 2], fs, 'TRI', people)
        [emg_ANT, t3] = emg_rectification(emg[:, 3], fs, 'ANT', people)
        [emg_POS, t4] = emg_rectification(emg[:, 4], fs, 'POS', people)
        [emg_PEC, t5] = emg_rectification(emg[:, 5], fs, 'PEC', people)
        [emg_LAT, t6] = emg_rectification(emg[:, 6], fs, 'LAT', people)

    # if label == 'chenzui-left-3kg' or label == 'chenzui-left-5.5kg' or label == 'chenzui-left-all-2.5kg' or label == 'chenzui-left-all-3kg' or label == 'chenzui-left-all-4kg' or label == 'chenzui-left-all-6.5kg':
    #     [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', 'chenzui')
    #     [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', 'chenzui')
    #     [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', 'chenzui')
    #     if include_TRI is True:
    #         [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', 'chenzui')
    # elif label == 'zhuo-left-1kg' or label == 'zhuo-right-3kg':
    #     [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', 'zhuo')
    #     [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', 'zhuo')
    #     [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', 'zhuo')
    #     if include_TRI is True:
    #         [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', 'zhuo')

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
    # elif label == 'chenzui-left-all-6.5kg-cts':
    #     t1 = t1 - 5.6085 + 12.617
    elif label == 'chenzui-left-all-4kg-cts':
        t1 = t1 - 83.755 + 90.078
    elif label == 'chenzui-left-all-5.5kg-cts':
        t1 = t1 - 3.6855 + 10.216
    elif label == 'chenzui-left-all-6.5kg-cts':
        t1 = t1 - 4.2 + 10.717
    elif label == 'bp-chenzui-left-4kg':
        t1 = t1 - 10.266 + 5.916
    elif label == 'bp-zhuo-right-3kg':
        t1 = t1 - 12.699 + 7.61
    elif label == 'bp-zhuo-right-4kg':
        t1 = t1 - 12.9 + 4.749

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
        if sport_label == 'biceps_curl':
            emg[0].append(emg_BIC[idx])
            emg[1].append(emg_BRA[idx])
            emg[2].append(emg_BRD[idx])
            if include_TRI is True:
                emg[3].append(emg_TRI[idx])
            # emg[3].append(emg_TRIlong[idx])
            # emg[4][i].append(emg_TRIlat[idx])
        elif sport_label == 'bench_press':
            emg[0].append(emg_BIC[idx])
            emg[1].append(emg_TRI[idx])
            emg[2].append(emg_ANT[idx])
            emg[3].append(emg_POS[idx])
            emg[4].append(emg_PEC[idx])
            emg[5].append(emg_LAT[idx])

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
    emg_std = np.asarray(emg_std).squeeze()
    emg = np.asarray(emg).squeeze()
    time_emg = np.asarray(time_emg).squeeze()

    output = {'emg_mean': emg_mean_out,
              'emg_std': emg_std_out,
              'arm': arm_out,
              'torque': tor_out,
              'time': t_tor_out,
              'emg_mean_long': emg_mean,
              'emg_std_long': emg_std,
              'emg': emg,
              'time_emg': time_emg,
              'trend_u': emg_trend_u_out,
              'trend_d': emg_trend_d_out}
    # return emg_mean_out, emg_std_out, arm_out, fa_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    # return emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    return output


def read_realted_files_bp(label='chenzui-left-3kg', idx='1', include_state=include_state):
    '''
    :param label:
    :param idx:
    :return: emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    '''
    fs = 1000
    ID = 'inverse_dynamics-'
    JA = 'Xsens_jointangle_q-'
    LE = 'MuscleAnalysis_Length-'
    TL = 'MuscleAnalysis_TendonLength-'
    if left_or_right == 'left':
        MA = 'modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-'
        MA1 = 'Subject-test-scaled_MuscleAnalysis_MomentArm_arm_flex_l-'
        MA2 = 'Subject-test-scaled_MuscleAnalysis_MomentArm_elbow_flex_l-'
    else:
        MA = 'MuscleAnalysis_MomentArm_elbow_flex_r-'
        MA1 = 'MuscleAnalysis_MomentArm_arm_flex_r-'
        MA2 = 'MuscleAnalysis_MomentArm_elbow_flex_r-'

    if label == 'bp-chenzui-left-4kg':
        people = 'chenzui'
        assert idx in ['1', '2', '3', '4', '5', '6']
        file_folder = 'files/bench press/chenzui/'
        emg = np.load(file_folder + '4' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '4' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '4' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '4' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '4' + '.xlsx')
        emg_mean = np.load('emg/bp-chenzui_mean_4kg.npy')
        emg_std = np.load('emg/bp-chenzui_std_4kg.npy')
        emg_trend_u = np.load('emg/bp-chenzui_trend_u_4kg.npy')
        emg_trend_d = np.load('emg/bp-chenzui_trend_d_4kg.npy')
        t_delta_joi = 10.266
        t_delta_emg = 5.916
        if idx == '1':
            timestep_emg = [21.999, 26.499, 26.499, 27.898]
        elif idx == '2':
            timestep_emg = [28.632, 32.732, 32.732, 34.031]
        elif idx == '3':
            timestep_emg = [34.498, 38.031, 38.031, 39.365]
        elif idx == '4':
            timestep_emg = [46.797, 50.997, 50.997, 53.364]
        elif idx == '5':
            timestep_emg = [54.164, 57.564, 57.564, 60.33]
        elif idx == '6':
            timestep_emg = [60.33, 64.363, 64.363, 67.296]
    elif label == 'bp-chenzui-left-5.5kg':
        people = 'chenzui'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        assert idx in idx_list
        file_folder = 'files/bench press/chenzui/'
        emg = np.load(file_folder + '5.5' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '5.5' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '5.5' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '5.5' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '5.5' + '.xlsx')
        emg_mean = np.load('emg/bp-chenzui_mean_5.5kg.npy')
        emg_std = np.load('emg/bp-chenzui_std_5.5kg.npy')
        emg_trend_u = np.load('emg/bp-chenzui_trend_u_5.5kg.npy')
        emg_trend_d = np.load('emg/bp-chenzui_trend_d_5.5kg.npy')
        t_delta_emg = 5.655
        t_delta_joi = 6.467
        timestep = [
            [16.266, 20.099, 20.099, 22.266],
            [22.699, 26.899, 26.899, 29.432],
            [29.432, 34.882, 34.882, 37.298],
            [37.932, 42.365, 42.365, 44.565],
            [45.131, 49.164, 49.164, 51.664],
            [52.131, 56.097, 56.097, 58.697],
            [58.697, 62.897, 62.897, 65.097],
            [65.497, 69.00, 69.00, 71.296],
            [72.596, 75.00, 75.00, 77.463],
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-chenzui-left-6.5kg':
        people = 'chenzui'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8']
        assert idx in idx_list
        file_folder = 'files/bench press/chenzui/'
        emg = np.load(file_folder + '6.5' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '6.5' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '6.5' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '6.5' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '6.5' + '.xlsx')
        emg_mean = np.load('emg/bp-chenzui_mean_6.5kg.npy')
        emg_std = np.load('emg/bp-chenzui_std_6.5kg.npy')
        emg_trend_u = np.load('emg/bp-chenzui_trend_u_6.5kg.npy')
        emg_trend_d = np.load('emg/bp-chenzui_trend_d_6.5kg.npy')
        t_delta_emg = 6.8905
        t_delta_joi = 5.833
        timestep = [
            [13.533, 16.166, 16.166, 18.066],
            [18.533, 21.766, 21.766, 23.866],
            [24.333, 27.632, 27.632, 29.999],
            [30.199, 34.299, 34.299, 36.132],
            [36.132, 40.165, 40.165, 42.398],
            [42.998, 46.265, 46.265, 49.098],
            [49.565, 52.964, 52.964, 55.264],
            [55.831, 58.764, 58.764, 61.164],
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-chenzui-left-7kg':
        people = 'chenzui'
        idx_list = ['1', '2', '3', '4', '5', '6']
        assert idx in idx_list
        file_folder = 'files/bench press/chenzui/'
        emg = np.load(file_folder + '7' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '7' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '7' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '7' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '7' + '.xlsx')
        emg_mean = np.load('emg/bp-chenzui_mean_7kg.npy')
        emg_std = np.load('emg/bp-chenzui_std_7kg.npy')
        emg_trend_u = np.load('emg/bp-chenzui_trend_u_7kg.npy')
        emg_trend_d = np.load('emg/bp-chenzui_trend_d_7kg.npy')
        t_delta_emg = 5.6485
        t_delta_joi = 6.1
        timestep = [
            [15.566, 18.732, 18.732, 20.532],
            [20.532, 24.065, 24.065, 25.899],
            # [26.232, 29.966, 29.966, 32.065],
            [32.199, 35.832, 35.832, 38.132],
            [38.132, 42.065, 42.065, 44.598],
            [44.931, 48.265, 48.265, 50.531],
            [50.531, 54.831, 54.831, 56.731],
            # [57.197, 61.047, 61.047, 63.063],
            # [63.63, 66.63, 66.63, 69.13],
            # [69.43, 73.497, 73.497, 75.496]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-chenzui-left-9.5kg':
        people = 'chenzui'
        idx_list = ['1', '2', '3', '4', '5', '6', '7']
        assert idx in idx_list
        file_folder = 'files/bench press/chenzui/'
        emg = np.load(file_folder + '9.5' + '.npy')
        joint_angle = pd.read_excel(file_folder + JA + '9.5' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '9.5' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '9.5' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '9.5' + '.xlsx')
        emg_mean = np.load('emg/bp-chenzui_mean_9.5kg.npy')
        emg_std = np.load('emg/bp-chenzui_std_9.5kg.npy')
        emg_trend_u = np.load('emg/bp-chenzui_trend_u_9.5kg.npy')
        emg_trend_d = np.load('emg/bp-chenzui_trend_d_9.5kg.npy')
        t_delta_emg = 6.628
        t_delta_joi = 5.633
        timestep = [
            [16.699, 18.965, 18.965, 20.299],
            # [20.932, 23.465, 23.465, 24.865],
            [25.232, 27.765, 27.765, 29.131],
            [29.598, 32.065, 32.065, 33.531],
            [33.664, 36.764, 36.764, 38.231],
            [38.431, 41.464, 41.464, 42.897],
            # [43.597, 46.047, 46.047, 47.364],
            [48.064, 50.397, 50.397, 51.897],
            [53.064, 55.23, 55.23, 56.83],
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-20kg' and date == '240408':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/yuetian/0408/'
        emg = np.asarray(pd.read_excel(file_folder + 'Test EMG/bptest 2024_04_08 19_25_06.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '20' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '20' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '20' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '20' + '.xlsx')
        emg_mean = np.load(file_folder + 'Test EMG/mean_20kg.npy')
        emg_std = np.load(file_folder + 'Test EMG/std_20kg.npy')
        emg_trend_u = np.load(file_folder + 'Test EMG/trend_u_20kg.npy')
        emg_trend_d = np.load(file_folder + 'Test EMG/trend_d_20kg.npy')
        t_delta_emg = 5.1
        t_delta_joi = 3.3
        timestep = [
            [8.483, 9.8, 9.8, 10.616],
            [10.616, 11.583, 11.583, 12.316],
            [12.316, 13.3, 13.3, 13.983],
            [13.983, 14.699, 14.699, 15.316],
            [15.316, 15.949, 15.949, 16.466],
            [16.466, 17.149, 17.149, 17.749],
            [17.749, 18.366, 18.366, 18.916],
            [18.916, 19.499, 19.499, 20.083],
            [20.083, 20.666, 20.666, 21.199],
            [21.199, 21.749, 21.749, 22.332]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-30kg' and date == '240408':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8']
        assert idx in idx_list
        file_folder = 'files/bench press/yuetian/0408/'
        emg = np.asarray(pd.read_excel(file_folder + 'Test EMG/bptest 2024_04_08 19_29_21.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '30' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '30' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '30' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '30' + '.xlsx')
        emg_mean = np.load(file_folder + 'Test EMG/mean_30kg.npy')
        emg_std = np.load(file_folder + 'Test EMG/std_30kg.npy')
        emg_trend_u = np.load(file_folder + 'Test EMG/trend_u_30kg.npy')
        emg_trend_d = np.load(file_folder + 'Test EMG/trend_d_30kg.npy')
        # t_delta_emg = 4.816
        # t_delta_joi = 3.56
        t_delta_emg = 3.56
        t_delta_joi = 4.816
        timestep = [
            [9.482, 11.199, 11.199, 12.299],
            [12.699, 14.532, 14.532, 15.565],
            [16.249, 17.949, 17.949, 19.049],
            [19.499, 21.332, 21.332, 22.465],
            [22.898, 24.965, 24.965, 25.965],
            [26.198, 28.415, 28.415, 29.398],
            [29.631, 31.465, 31.465, 32.415],
            [32.648, 34.731, 34.731, 35.748]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-40kg' and date == '240408':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8']
        assert idx in idx_list
        file_folder = 'files/bench press/yuetian/0408/'
        emg = np.asarray(pd.read_excel(file_folder + '40' + '.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '40' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '40' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '40' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '40' + '.xlsx')
        emg_mean = np.load(file_folder + 'Test EMG/mean_40kg.npy')
        emg_std = np.load(file_folder + 'Test EMG/std_40kg.npy')
        emg_trend_u = np.load(file_folder + 'Test EMG/trend_u_40kg.npy')
        emg_trend_d = np.load(file_folder + 'Test EMG/trend_d_40kg.npy')
        t_delta_emg = 3.217
        t_delta_joi = 5.633
        timestep = [
            [11.616, 13.866, 13.866, 14.849],
            [15.599, 17.283, 17.283, 18.133],
            [18.633, 20.266, 20.266, 21.182],
            [21.632, 23.282, 23.282, 24.216],
            [24.399, 26.116, 26.116, 26.999],
            [27.199, 28.749, 28.749, 29.515],
            [29.882, 31.249, 31.249, 32.049],
            [32.049, 33.499, 33.499, 34.798]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-50kg' and date == '240408':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8']
        assert idx in idx_list
        file_folder = 'files/bench press/yuetian/0408/'
        emg = np.asarray(pd.read_excel(file_folder + '50' + '.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '50' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '50' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '50' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '50' + '.xlsx')
        emg_mean = np.load(file_folder + 'Test EMG/mean_50kg.npy')
        emg_std = np.load(file_folder + 'Test EMG/std_50kg.npy')
        emg_trend_u = np.load(file_folder + 'Test EMG/trend_u_50kg.npy')
        emg_trend_d = np.load(file_folder + 'Test EMG/trend_d_50kg.npy')
        t_delta_emg = 2.23
        t_delta_joi = 4.617
        timestep = [
            [10.566, 12.333, 12.333, 13.25],
            [13.583, 15.233, 15.233, 16.183],
            [16.433, 17.566, 17.566, 18.599],
            [18.833, 19.899, 19.899, 20.833],
            [21.099, 21.966, 21.966, 23.099],
            [23.099, 24.249, 24.249, 25.416],
            [25.599, 26.899, 26.899, 27.666],
            [27.982, 29.216, 29.216, 30.099]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-60kg' and date == '240408':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-6/yuetian/0408/'
        emg = np.asarray(pd.read_excel(file_folder + '60' + '.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '60' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '60' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '60' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '60' + '.xlsx')
        emg_mean = np.load(file_folder + 'Test EMG/mean_60kg.npy')
        emg_std = np.load(file_folder + 'Test EMG/std_60kg.npy')
        emg_trend_u = np.load(file_folder + 'Test EMG/trend_u_60kg.npy')
        emg_trend_d = np.load(file_folder + 'Test EMG/trend_d_60kg.npy')
        t_delta_emg = 3.273
        t_delta_joi = 5.317
        timestep = [
            [10.216, 12.25, 12.25, 13.099],
            [13.349, 14.183, 14.183, 15.149],
            [15.399, 16.349, 16.349, 17.199],
            [17.366, 18.449, 18.449, 19.366],
            [19.599, 20.432, 20.432, 21.532],
            [21.632, 22.466, 22.466, 23.316],
            [23.316, 24.416, 24.416, 25.349],
            [25.349, 26.249, 26.249, 27.149],
            [27.382, 28.165, 28.165, 29.015],
            [29.015, 30.048, 30.048, 30.982]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-20kg' and date == '240604':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_06_04 16_16_22.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '20' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '20' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '20' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '20' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [6.650, 8.133, 8.133, 8.983],
            [9.583, 10.766, 10.766, 11.583],
            [12.032, 13.149, 13.149, 13.949],
            [14.316, 15.599, 15.599, 16.549],
            [16.732, 17.956, 17.956, 19.032],
            [19.682, 20.715, 20.715, 21.449],
            [22.015, 23.082, 23.082, 23.865],
            [23.998, 25.315, 25.315, 26.182],
            [26.565, 27.698, 27.698, 28.615],
            [28.982, 30.132, 30.132, 31.064]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-30kg' and date == '240604':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_06_04 16_21_47.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '30' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '30' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '30' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '30' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [11.449, 12.633, 12.633, 13.383],
            [13.449, 14.399, 14.399, 15.149],
            [15.366, 16.182, 16.182, 16.882],
            [17.032, 17.916, 17.916, 18.749],
            [18.749, 19.566, 19.566, 20.249],
            [20.315, 21.032, 21.032, 21.682],
            [21.682, 22.415, 22.415, 23.115],
            [23.115, 23.765, 23.765, 24.482],
            [24.482, 25.215, 25.215, 25.915],
            [25.915, 26.765, 26.765, 27.365]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-40kg' and date == '240604':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_06_04 16_23_42.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '40' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '40' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '40' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '40' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [7.600, 8.783, 8.783, 9.517],
            [9.683, 10.483, 10.483, 11.117],
            [11.200, 11.867, 11.867, 12.500],
            [12.567, 13.300, 13.300, 13.900],
            [13.900, 14.550, 14.550, 15.183],
            [15.216, 16.200, 16.200, 16.783],
            [16.783, 17.383, 17.383, 17.949],
            [17.949, 18.533, 18.533, 19.100],
            [19.100, 19.683, 19.683, 20.216],
            [20.216, 20.833, 20.833, 21.433]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-50kg' and date == '240604':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_06_04 16_26_07.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '50' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '50' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '50' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '50' + '.xlsx')
        # emg_mean = np.load(file_folder + 'emg/mean_50kg.npy')
        # emg_std = np.load(file_folder + 'emg/std_50kg.npy')
        # emg_trend_u = np.load(file_folder + 'emg/trend_u_50kg.npy')
        # emg_trend_d = np.load(file_folder + 'emg/trend_d_50kg.npy')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [7.850, 8.783, 8.783, 9.350],
            [9.666, 10.250, 10.250, 10.800],
            [11.066, 11.650, 11.650, 12.216],
            [12.400, 13.000, 13.000, 13.566],
            [13.733, 14.266, 14.266, 14.849],
            [14.916, 15.533, 15.533, 16.216],
            [16.216, 16.816, 16.816, 17.399],
            [17.499, 18.033, 18.033, 18.599],
            [18.666, 19.183, 19.183, 19.783],
            [19.783, 20.333, 20.333, 21.116]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-yuetian-right-60kg' and date == '240604':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_06_04 16_29_21.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '60' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '60' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '60' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '60' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [22.365, 23.249, 23.249, 23.915],
            [23.915, 24.599, 24.599, 25.299],
            [25.299, 26.132, 26.132, 26.982],
            [27.115, 27.865, 27.865, 28.648],
            [28.715, 29.398, 29.398, 30.165],
            [30.165, 30.815, 30.815, 31.582],
            [31.732, 32.315, 32.315, 33.082],
            [33.082, 33.715, 33.715, 34.381],
            [34.381, 35.015, 35.015, 35.698],
            [35.698, 36.298, 36.298, 36.998]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-kehan-right-20kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/kehan/'
        emg = from_csv_to_emg(file_folder + 'emg/test 2024_07_31 11_25_10.csv')
        joint_angle = pd.read_excel(file_folder + JA + '20' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '20' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '20' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '20' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [6.650, 8.133, 8.133, 8.983],
            [9.583, 10.766, 10.766, 11.583],
            [12.032, 13.149, 13.149, 13.949],
            [14.316, 15.599, 15.599, 16.549],
            [16.732, 17.956, 17.956, 19.032],
            [19.682, 20.715, 20.715, 21.449],
            [22.015, 23.082, 23.082, 23.865],
            [23.998, 25.315, 25.315, 26.182],
            [26.565, 27.698, 27.698, 28.615],
            [28.982, 30.132, 30.132, 31.064]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-kehan-right-30kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/kehan/'
        emg = from_csv_to_emg(file_folder + 'emg/test 2024_07_31 11_27_28.csv')
        joint_angle = pd.read_excel(file_folder + JA + '30' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '30' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '30' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '30' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [10.466, 11.616, 11.616, 12.283],
            [13.083, 14.100, 14.100, 14.666],
            [15.399, 16.516, 16.516, 17.066],
            [17.766, 18.749, 18.749, 19.266],
            [19.966, 21.099, 21.099, 21.649],
            [22.616, 23.682, 23.682, 24.216],
            [25.149, 26.199, 26.199, 26.732],
            [27.466, 28.482, 28.482, 29.032],
            [29.799, 30.849, 30.849, 31.399],
            [32.165, 33.182, 33.182, 33.732]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-kehan-right-40kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/kehan/'
        emg = from_csv_to_emg(file_folder + 'emg/test 2024_07_31 11_29_38.csv')
        joint_angle = pd.read_excel(file_folder + JA + '40' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '40' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '40' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '40' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [16.866, 17.966, 17.966, 18.566],
            [19.132, 19.949, 19.949, 20.516],
            [21.066, 22.182, 22.182, 22.815],
            [23.215, 24.315, 24.315, 24.932],
            [25.465, 26.615, 26.615, 27.265],
            [27.765, 28.849, 28.849, 29.498],
            [30.032, 31.082, 31.082, 31.765],
            [32.448, 33.448, 33.448, 34.148]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-kehan-right-45kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6', '7', '8']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/kehan/'
        emg = from_csv_to_emg(file_folder + 'emg/test 2024_07_31 11_42_15.csv')
        joint_angle = pd.read_excel(file_folder + JA + '45' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '45' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '45' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '45' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [9.083, 10.066, 10.066, 10.716],
            [11.366, 12.183, 12.183, 12.749],
            [13.599, 14.416, 14.416, 15.066],
            [16.099, 17.049, 17.049, 17.733],
            [18.616, 19.482, 19.482, 20.166],
            [20.849, 21.732, 21.732, 22.499],
            [23.316, 24.299, 24.299, 25.165],
            [25.999, 27.082, 27.082, 27.982]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'bp-kehan-right-50kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5']
        assert idx in idx_list
        file_folder = 'files/bench press/muscle-18/kehan/'
        emg = from_csv_to_emg(file_folder + 'emg/test 2024_07_31 11_36_27.csv')
        joint_angle = pd.read_excel(file_folder + JA + '50' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '50' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '50' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '50' + '.xlsx')
        emg_mean = np.load(file_folder + 'emg/mean_all.npy')
        emg_std = np.load(file_folder + 'emg/std_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [7.717, 9.017, 9.017, 9.767],
            [10.400, 11.483, 11.483, 12.266],
            [12.916, 14.100, 14.100, 14.950],
            [15.583, 16.750, 16.750, 17.533],
            [18.149, 19.466, 19.466, 20.449]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    else:
        print('No corresponding label!')
        return 0

    time_torque = moment['time']
    time_momarm = momarm1['time']
    time_angle = joint_angle['time']
    if left_or_right == 'left':
        angle1 = joint_angle['arm_flex_l']
        angle2 = joint_angle['elbow_flex_l']
        torque1 = moment['arm_flex_l_moment']
        torque2 = moment['elbow_flex_l_moment']
    else:
        angle1 = joint_angle['arm_flex_r']
        angle2 = joint_angle['elbow_flex_r']
        torque1 = moment['arm_flex_r_moment']
        torque2 = moment['elbow_flex_r_moment']

    torque1 = scipy.signal.savgol_filter(torque1, 53, 3)
    torque2 = scipy.signal.savgol_filter(torque2, 53, 3)

    t_tor = []
    t_arm = []
    t_ang = []
    tor1 = []
    tor2 = []
    ang1 = []
    ang2 = []
    arm1 = [[] for _ in range(len(muscle_idx))]
    arm2 = [[] for _ in range(len(muscle_idx))]
    ml = [[] for _ in range(len(muscle_idx))]
    tl = [[] for _ in range(len(muscle_idx))]
    fa = [[] for _ in range(len(muscle_idx))]
    fr = []

    for i in range(int(len(timestep_emg) / 2)):
        tas = find_nearest_idx(time_angle, timestep_emg[2 * i])
        tae = find_nearest_idx(time_angle, timestep_emg[2 * i + 1])
        t_ang.append(resample_by_len(list(time_angle[tas:tae]), target_len))
        ang1.append(resample_by_len(list(angle1[tas:tae]), target_len))
        ang2.append(resample_by_len(list(angle2[tas:tae]), target_len))

        tts = find_nearest_idx(time_torque, timestep_emg[2 * i])
        tte = find_nearest_idx(time_torque, timestep_emg[2 * i + 1])
        t_tor.append(resample_by_len(list(time_torque[tts:tte]), target_len))
        tor1.append(resample_by_len(list(torque1[tts:tte]), target_len))
        tor2.append(resample_by_len(list(torque2[tts:tte]), target_len))

        tms = find_nearest_idx(time_momarm, timestep_emg[2 * i])
        tme = find_nearest_idx(time_momarm, timestep_emg[2 * i + 1])
        t_arm.append(resample_by_len(list(time_momarm[tms:tme]), target_len))
        for j in range(len(muscle_idx)):
            if arm_constant is True:
                marm1 = np.ones_like(momarm1[muscle_idx[j]][tms:tme]) * np.mean(momarm1[muscle_idx[j]][tms:tme])
                marm2 = np.ones_like(momarm2[muscle_idx[j]][tms:tme]) * np.mean(momarm2[muscle_idx[j]][tms:tme])
                arm1[j].append(resample_by_len(list(marm1), target_len))
                arm2[j].append(resample_by_len(list(marm2), target_len))
            else:
                arm1[j].append(resample_by_len(list(momarm1[muscle_idx[j]][tms:tme]), target_len))
                arm2[j].append(resample_by_len(list(momarm2[muscle_idx[j]][tms:tme]), target_len))
            # ml[j].append(resample_by_len(list(length[muscle_idx[j]][tms:tme]), target_len))
            # tl[j].append(resample_by_len(list(tenlen[muscle_idx[j]][tms:tme]), target_len))
            # for k in range(tms, tme):
            #     fr.append(calc_fal(length[muscle_idx[j]][k], tenlen[muscle_idx[j]][k], muscle_idx[j]))
            # fa[j].append(resample_by_len(fr, target_len))
            fr = []

    t_tor_out = []
    t_arm_out = []
    t_ang_out = []
    emg_mean_out = []
    emg_std_out = []
    emg_trend_u_out = []
    emg_trend_d_out = []
    tor_out1 = []
    tor_out2 = []
    ang_out1 = []
    ang_out2 = []
    arm_out1 = [[] for _ in range(len(muscle_idx))]
    arm_out2 = [[] for _ in range(len(muscle_idx))]
    ml_out = [[] for _ in range(len(muscle_idx))]
    tl_out = [[] for _ in range(len(muscle_idx))]
    fa_out = [[] for _ in range(len(muscle_idx))]
    for i in range(len(muscle_idx)):
        if include_state == 'lift and down':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :], target_len * 2))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :], target_len * 2))
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len * 2))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len * 2))
            arm_out1[i].append(np.concatenate([arm1[i][0], arm1[i][1]]))
            arm_out2[i].append(np.concatenate([arm2[i][0], arm2[i][1]]))
            # ml_out[i].append(np.concatenate([ml[i][0], ml[i][1]]))
            # tl_out[i].append(np.concatenate([tl[i][0], tl[i][1]]))
            # fa_out[i].append(np.concatenate([fa[i][0], fa[i][1]]))
        elif include_state == 'lift':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :emg_lift_len], target_len))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :emg_lift_len], target_len))
            # emg_mean = emg_mean[:, :emg_lift_len]
            # emg_std = emg_std[:, :emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :emg_lift_len], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :emg_lift_len], target_len))
            arm_out1[i].append(arm1[i][0])
            arm_out2[i].append(arm2[i][0])
            # ml_out[i].append(ml[i][0])
            # tl_out[i].append(tl[i][0])
            # fa_out[i].append(fa[i][0])
        elif include_state == 'down':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, 0:emg_lift_len], target_len))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, 0:emg_lift_len], target_len))
            emg_mean = emg_mean[:, 0:emg_lift_len]
            emg_std = emg_std[:, 0:emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len))
            arm_out1[i].append(arm1[i][1])
            arm_out2[i].append(arm2[i][1])
            # ml_out[i].append(ml[i][1])
            # tl_out[i].append(tl[i][1])
            # fa_out[i].append(fa[i][1])
        else:
            return os.error('State must include lift or down!')

    if include_state == 'lift and down':
        t_tor_out.append(np.concatenate([t_tor[0], t_tor[1]]))
        t_arm_out.append(np.concatenate([t_arm[0], t_arm[1]]))
        t_ang_out.append(np.concatenate([t_ang[0], t_ang[1]]))
        tor_out1.append(np.concatenate([tor1[0], tor1[1]]))
        tor_out2.append(np.concatenate([tor2[0], tor2[1]]))
        ang_out1.append(np.concatenate([ang1[0], ang1[1]]))
        ang_out2.append(np.concatenate([ang2[0], ang2[1]]))
    elif include_state == 'lift':
        t_tor_out.append(t_tor[0])
        t_arm_out.append(t_arm[0])
        t_ang_out.append(t_ang[0])
        tor_out1.append(tor1[0])
        tor_out2.append(tor2[0])
        ang_out1.append(ang1[0])
        ang_out2.append(ang2[0])
    elif include_state == 'down':
        t_tor_out.append(t_tor[1])
        t_arm_out.append(t_arm[1])
        t_ang_out.append(t_ang[1])
        tor_out1.append(tor1[1])
        tor_out2.append(tor2[1])
        ang_out1.append(ang1[1])
        ang_out2.append(ang2[1])

    if sport_label == 'biceps_curl':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people, left)
        [emg_BRA, t2] = emg_rectification(emg[:, 2], fs, 'BRA', people, left)
        [emg_BRD, t3] = emg_rectification(emg[:, 3], fs, 'BRD', people, left)
        if include_TRI is True:
            [emg_TRI, t4] = emg_rectification(emg[:, 4], fs, 'TRI', people, left)
    elif sport_label == 'bench_press':
        if date == '240408':
            [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people, left)
            [emg_TRI, t2] = emg_rectification(emg[:, 2], fs, 'TRI', people, left)
            [emg_ANT, t3] = emg_rectification(emg[:, 3], fs, 'ANT', people, left)
            [emg_POS, t4] = emg_rectification(emg[:, 4], fs, 'POS', people, left)
            [emg_PEC, t5] = emg_rectification(emg[:, 5], fs, 'PEC', people, left)
            [emg_LAT, t6] = emg_rectification(emg[:, 6], fs, 'LAT', people, left)
        else:
            emg_rect_label = musc_label
            emg_list = []
            for i in range(len(emg_rect_label)):
                if joint_idx == 'shoulder':
                    if i < 8:
                        [emg_rect, t1] = emg_rectification(emg[:, i + 1], fs, emg_rect_label[i], people, left)
                    else:
                        [emg_rect, t1] = emg_rectification(emg[:, i + 4], fs, emg_rect_label[i], people, left)
                elif joint_idx == 'elbow':
                    [emg_rect, t1] = emg_rectification(emg[:, i + 7], fs, emg_rect_label[i], people, left)
                else:
                    [emg_rect, t1] = emg_rectification(emg[:, i + 1], fs, emg_rect_label[i], people, left)
                emg_list.append(emg_rect)

    if label == 'bp-chenzui-left-4kg':
        t1 = t1 - 5.916 + 10.266
    else:
        t1 = t1 - t_delta_emg + t_delta_joi

    allk = related_muscle_num
    alli = related_muscle_idx

    if need_all_muscle is True:
        emg = [([]) for _ in range(len(muscle_idx))]
    else:
        emg = [([]) for _ in range(len(measured_muscle_idx))]
    time_emg = [[]]
    for t in t_tor_out[0]:
        idx = find_nearest_idx(t1, t)
        time_emg[0].append(t1[idx])
        if sport_label == 'biceps_curl':
            emg[0].append(emg_BIC[idx])
            emg[1].append(emg_BRA[idx])
            emg[2].append(emg_BRD[idx])
            if include_TRI is True:
                emg[3].append(emg_TRI[idx])
            # emg[3].append(emg_TRIlong[idx])
            # emg[4][i].append(emg_TRIlat[idx])
        elif sport_label == 'bench_press':
            if date == '240408':
                emg[0].append(emg_BIC[idx])
                emg[1].append(emg_TRI[idx])
                emg[2].append(emg_ANT[idx])
                emg[3].append(emg_POS[idx])
                emg[4].append(emg_PEC[idx])
                emg[5].append(emg_LAT[idx])
                if need_all_muscle is True:
                    if elbow_muscle is True:
                        # emg[6].append(emg_BIC[idx])
                        for k in range(11):
                            emg[6 + k].append(emg_BIC[idx])
                        for k in range(3):
                            emg[17 + k].append(emg_TRI[idx])
                        for k in range(3):
                            emg[20 + k].append(emg_ANT[idx])
                        for k in range(7):
                            emg[23 + k].append(emg_POS[idx])
                        for k in range(5):
                            emg[30 + k].append(emg_PEC[idx])
                        for k in range(5):
                            emg[35 + k].append(emg_LAT[idx])
                        # a1, a2 = elbow_emg(emg_BIC[idx], emg_TRI[idx])
                        # for j in range(3):
                        #     emg[14 + j].append(a1)  # brachiorad
                        # for j in range(7):
                        #     emg[7 + j].append(a2)  # brachialis
                    else:
                        for k in range(3):
                            emg[6 + k].append(emg_TRI[idx])
                        for k in range(3):
                            emg[9 + k].append(emg_ANT[idx])
                        for k in range(7):
                            emg[12 + k].append(emg_POS[idx])
                        for k in range(5):
                            emg[19 + k].append(emg_PEC[idx])
                        for k in range(5):
                            emg[24 + k].append(emg_LAT[idx])
            else:
                for i in range(len(measured_muscle_idx)):
                    emg[i].append(emg_list[i][idx])
                if need_all_muscle is True:
                    for i in range(len(measured_muscle_idx)):
                        for k in range(allk[i]):
                            emg[alli[i] + k].append(emg_list[i][idx])

    emg_trend_u_out = np.asarray(emg_trend_u_out).squeeze()
    emg_trend_d_out = np.asarray(emg_trend_d_out).squeeze()
    emg_mean_out = np.asarray(emg_mean_out).squeeze()
    emg_std_out = np.asarray(emg_std_out).squeeze()
    arm_out1 = np.asarray(arm_out1).squeeze()
    arm_out2 = np.asarray(arm_out2).squeeze()
    # fa_out = np.asarray(fa_out).squeeze()
    tor_out1 = np.asarray(tor_out1).squeeze()
    tor_out2 = np.asarray(tor_out2).squeeze()
    ang_out1 = np.asarray(ang_out1).squeeze()
    ang_out2 = np.asarray(ang_out2).squeeze()
    t_tor_out = np.asarray(t_tor_out).squeeze()
    emg_mean = np.asarray(emg_mean).squeeze()
    emg_std = np.asarray(emg_std).squeeze()
    emg = np.asarray(emg).squeeze()
    time_emg = np.asarray(time_emg).squeeze()

    arm = np.dstack([arm_out1, arm_out2])
    torque = np.column_stack((tor_out1, tor_out2))

    output = {'emg_mean': emg_mean_out,
              'emg_std': emg_std_out,
              'arm': arm,
              'arm1': arm_out1,
              'arm2': arm_out2,
              'torque': torque,
              'torque1': tor_out1,
              'torque2': tor_out2,
              'angle1': ang_out1,
              'angle2': ang_out2,
              'time': t_tor_out,
              'emg_mean_long': emg_mean,
              'emg_std_long': emg_std,
              'emg': emg,
              'time_emg': time_emg,
              'trend_u': emg_trend_u_out,
              'trend_d': emg_trend_d_out}
    return output


def read_realted_files_dl(label='dl-yuetian-right-35kg', idx='1', include_state=include_state):
    '''
    :param label:
    :param idx:
    :return: emg_mean_out, emg_std_out, arm_out, tor_out, t_tor_out, emg_mean, emg_std, emg, time_emg, emg_trend_u_out, emg_trend_d_out
    '''
    fs = 1000
    ID = 'inverse_dynamics-'
    JA = 'Xsens_jointangle_q-'
    LE = 'modelModified_MuscleAnalysis_Length-'
    TL = 'modelModified_MuscleAnalysis_TendonLength-'
    if left_or_right == 'left':
        MA = 'modelModified_MuscleAnalysis_MomentArm_elbow_flex_l-'
        MA1 = 'WholeBodyModel-yuetian_MuscleAnalysis_MomentArm_flex_extension-'
        MA2 = 'WholeBodyModel-yuetian_MuscleAnalysis_MomentArm_hip_flexion_l-'
        MA3 = 'WholeBodyModel-yuetian_MuscleAnalysis_MomentArm_knee_angle_l-'
    else:
        MA = 'modelModified_MuscleAnalysis_MomentArm_elbow_flex_r-'
        MA1 = 'WholeBodyModel-yuetian_MuscleAnalysis_MomentArm_flex_extension-'
        MA2 = 'WholeBodyModel-yuetian_MuscleAnalysis_MomentArm_hip_flexion_r-'
        MA3 = 'WholeBodyModel-yuetian_MuscleAnalysis_MomentArm_knee_angle_r-'

    if label == 'dl-yuetian-right-35kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6']
        assert idx in idx_list
        file_folder = 'files/deadlift/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_05_17 16_27_52.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '35' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '35' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '35' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '35' + '.xlsx')
        momarm3 = pd.read_excel(file_folder + MA3 + '35' + '.xlsx')
        # emg_mean = np.load(file_folder + 'emg/mean_dl_35kg.npy')
        # emg_std = np.load(file_folder + 'emg/std_dl_35kg.npy')
        # emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_35kg.npy')
        # emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_35kg.npy')
        emg_mean = np.load(file_folder + 'emg/mean_dl_all.npy')
        emg_std = np.load(file_folder + 'emg/std_dl_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [7.649, 8.949, 8.949, 10.532],
            [11.149, 12.333, 12.333, 14.099],
            [14.632, 15.665, 15.666, 17.215],
            [17.799, 18.898, 18.898, 20.566],
            [20.933, 21.933, 21.933, 23.366],
            [23.933, 25.049, 25.049, 26.132]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'dl-yuetian-right-45kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6']
        assert idx in idx_list
        file_folder = 'files/deadlift/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_05_17 16_30_59.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '45' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '45' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '45' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '45' + '.xlsx')
        momarm3 = pd.read_excel(file_folder + MA3 + '45' + '.xlsx')
        # emg_mean = np.load(file_folder + 'emg/mean_dl_45kg.npy')
        # emg_std = np.load(file_folder + 'emg/std_dl_45kg.npy')
        # emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_45kg.npy')
        # emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_45kg.npy')
        emg_mean = np.load(file_folder + 'emg/mean_dl_all.npy')
        emg_std = np.load(file_folder + 'emg/std_dl_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [8.783, 9.5, 9.5, 10.65],
            [11.4, 11.983, 11.983, 13.15],
            [13.933, 14.833, 14.833, 15.999],
            [16.75, 17.75, 17.75, 18.633],
            [19.416, 20.366, 20.366, 21.149],
            [21.883, 22.533, 22.533, 23.566]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'dl-yuetian-right-65kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5']
        assert idx in idx_list
        file_folder = 'files/deadlift/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_05_17 16_37_38.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '65' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '65' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '65' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '65' + '.xlsx')
        momarm3 = pd.read_excel(file_folder + MA3 + '65' + '.xlsx')
        # emg_mean = np.load(file_folder + 'emg/mean_dl_65kg.npy')
        # emg_std = np.load(file_folder + 'emg/std_dl_65kg.npy')
        # emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_65kg.npy')
        # emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_65kg.npy')
        emg_mean = np.load(file_folder + 'emg/mean_dl_all.npy')
        emg_std = np.load(file_folder + 'emg/std_dl_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [7.266, 8.166, 8.166, 9.316],
            [10.616, 11.466, 11.466, 12.732],
            [13.782, 14.466, 14.466, 15.682],
            [16.615, 17.399, 17.399, 18.349],
            [19.282, 20.039, 20.039, 21.049]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    elif label == 'dl-yuetian-right-75kg':
        people = 'yuetian'
        idx_list = ['1', '2', '3', '4', '5', '6']
        assert idx in idx_list
        file_folder = 'files/deadlift/yuetian/'
        emg = np.asarray(pd.read_excel(file_folder + 'emg/test 2024_05_17 16_41_21.xlsx'))
        joint_angle = pd.read_excel(file_folder + JA + '75' + '.xlsx')
        moment = pd.read_excel(file_folder + ID + '75' + '.xlsx')
        momarm1 = pd.read_excel(file_folder + MA1 + '75' + '.xlsx')
        momarm2 = pd.read_excel(file_folder + MA2 + '75' + '.xlsx')
        momarm3 = pd.read_excel(file_folder + MA3 + '75' + '.xlsx')
        # emg_mean = np.load(file_folder + 'emg/mean_dl_65kg.npy')
        # emg_std = np.load(file_folder + 'emg/std_dl_65kg.npy')
        # emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_65kg.npy')
        # emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_65kg.npy')
        emg_mean = np.load(file_folder + 'emg/mean_dl_all.npy')
        emg_std = np.load(file_folder + 'emg/std_dl_all.npy')
        emg_trend_u = np.load(file_folder + 'emg/trend_u_dl_all.npy')
        emg_trend_d = np.load(file_folder + 'emg/trend_d_dl_all.npy')
        t_delta_emg = 0
        t_delta_joi = 0
        timestep = [
            [6.566, 7.283, 7.283, 8.333],
            [9.133, 9.789, 9.789, 10.766],
            [11.483, 12.099, 12.099, 13.149],
            [13.916, 14.536, 14.536, 15.565],
            [16.199, 16.966, 16.966, 17.816],
            [18.482, 19.032, 19.032, 20.065]
        ]
        for i in range(len(idx_list)):
            if idx == idx_list[i]:
                timestep_emg = timestep[i]
                break
    else:
        print('No corresponding label!')
        return 0

    time_torque = moment['time']
    time_momarm = momarm1['time']
    time_angle = joint_angle['time']
    if left_or_right == 'left':
        angle1 = joint_angle['arm_flex_l']  # flex_extension
        angle2 = joint_angle['hip_flexion_l']
        angle3 = joint_angle['knee_angle_l']
        torque1 = moment['flex_extension_force']
        torque2 = moment['hip_flexion_l_moment']
        torque3 = moment['knee_angle_l_moment']
    else:
        angle1 = joint_angle['arm_flex_r']  # flex_extension
        angle2 = joint_angle['hip_flexion_r']
        angle3 = joint_angle['knee_angle_r']
        torque1 = moment['flex_extension_force']
        torque2 = moment['hip_flexion_r_moment']
        torque3 = moment['knee_angle_r_moment']

    torque1 = scipy.signal.savgol_filter(torque1, 53, 3)
    torque2 = scipy.signal.savgol_filter(torque2, 53, 3)
    torque3 = scipy.signal.savgol_filter(torque3, 53, 3)

    t_tor = []
    t_arm = []
    t_ang = []
    tor1 = []
    tor2 = []
    tor3 = []
    ang1 = []
    ang2 = []
    ang3 = []
    arm1 = [[] for _ in range(len(muscle_idx))]
    arm2 = [[] for _ in range(len(muscle_idx))]
    arm3 = [[] for _ in range(len(muscle_idx))]
    ml = [[] for _ in range(len(muscle_idx))]
    tl = [[] for _ in range(len(muscle_idx))]
    fa = [[] for _ in range(len(muscle_idx))]
    fr = []

    for i in range(int(len(timestep_emg) / 2)):
        tas = find_nearest_idx(time_angle, timestep_emg[2 * i])
        tae = find_nearest_idx(time_angle, timestep_emg[2 * i + 1])
        t_ang.append(resample_by_len(list(time_angle[tas:tae]), target_len))
        ang1.append(resample_by_len(list(angle1[tas:tae]), target_len))
        ang2.append(resample_by_len(list(angle2[tas:tae]), target_len))
        ang3.append(resample_by_len(list(angle3[tas:tae]), target_len))

        tts = find_nearest_idx(time_torque, timestep_emg[2 * i])
        tte = find_nearest_idx(time_torque, timestep_emg[2 * i + 1])
        t_tor.append(resample_by_len(list(time_torque[tts:tte]), target_len))
        tor1.append(resample_by_len(list(torque1[tts:tte]), target_len))
        tor2.append(resample_by_len(list(torque2[tts:tte]), target_len))
        tor3.append(resample_by_len(list(torque3[tts:tte]), target_len))

        tms = find_nearest_idx(time_momarm, timestep_emg[2 * i])
        tme = find_nearest_idx(time_momarm, timestep_emg[2 * i + 1])
        t_arm.append(resample_by_len(list(time_momarm[tms:tme]), target_len))
        for j in range(len(muscle_idx)):
            if arm_constant is True:
                marm1 = np.ones_like(momarm1[muscle_idx[j]][tms:tme]) * np.mean(momarm1[muscle_idx[j]][tms:tme])
                marm2 = np.ones_like(momarm2[muscle_idx[j]][tms:tme]) * np.mean(momarm2[muscle_idx[j]][tms:tme])
                marm3 = np.ones_like(momarm3[muscle_idx[j]][tms:tme]) * np.mean(momarm3[muscle_idx[j]][tms:tme])
                arm1[j].append(resample_by_len(list(marm1), target_len))
                arm2[j].append(resample_by_len(list(marm2), target_len))
                arm3[j].append(resample_by_len(list(marm3), target_len))
            else:
                arm1[j].append(resample_by_len(list(momarm1[muscle_idx[j]][tms:tme]), target_len))
                arm2[j].append(resample_by_len(list(momarm2[muscle_idx[j]][tms:tme]), target_len))
                arm3[j].append(resample_by_len(list(momarm3[muscle_idx[j]][tms:tme]), target_len))
            # ml[j].append(resample_by_len(list(length[muscle_idx[j]][tms:tme]), target_len))
            # tl[j].append(resample_by_len(list(tenlen[muscle_idx[j]][tms:tme]), target_len))
            # for k in range(tms, tme):
            #     fr.append(calc_fal(length[muscle_idx[j]][k], tenlen[muscle_idx[j]][k], muscle_idx[j]))
            # fa[j].append(resample_by_len(fr, target_len))
            fr = []

    t_tor_out = []
    t_arm_out = []
    t_ang_out = []
    emg_mean_out = []
    emg_std_out = []
    emg_trend_u_out = []
    emg_trend_d_out = []
    tor_out1 = []
    tor_out2 = []
    tor_out3 = []
    ang_out1 = []
    ang_out2 = []
    ang_out3 = []
    arm_out1 = [[] for _ in range(len(muscle_idx))]
    arm_out2 = [[] for _ in range(len(muscle_idx))]
    arm_out3 = [[] for _ in range(len(muscle_idx))]
    ml_out = [[] for _ in range(len(muscle_idx))]
    tl_out = [[] for _ in range(len(muscle_idx))]
    fa_out = [[] for _ in range(len(muscle_idx))]
    for i in range(len(muscle_idx)):
        if include_state == 'lift and down':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :], target_len * 2))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :], target_len * 2))
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len * 2))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len * 2))
            arm_out1[i].append(np.concatenate([arm1[i][0], arm1[i][1]]))
            arm_out2[i].append(np.concatenate([arm2[i][0], arm2[i][1]]))
            arm_out3[i].append(np.concatenate([arm3[i][0], arm3[i][1]]))
            # ml_out[i].append(np.concatenate([ml[i][0], ml[i][1]]))
            # tl_out[i].append(np.concatenate([tl[i][0], tl[i][1]]))
            # fa_out[i].append(np.concatenate([fa[i][0], fa[i][1]]))
        elif include_state == 'lift':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, :emg_lift_len], target_len))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, :emg_lift_len], target_len))
            # emg_mean = emg_mean[:, :emg_lift_len]
            # emg_std = emg_std[:, :emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :emg_lift_len], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :emg_lift_len], target_len))
            arm_out1[i].append(arm1[i][0])
            arm_out2[i].append(arm2[i][0])
            arm_out3[i].append(arm3[i][0])
            # ml_out[i].append(ml[i][0])
            # tl_out[i].append(tl[i][0])
            # fa_out[i].append(fa[i][0])
        elif include_state == 'down':
            emg_trend_u_out.append(resample_by_len(emg_trend_u[i, 0:emg_lift_len], target_len))
            emg_trend_d_out.append(resample_by_len(emg_trend_d[i, 0:emg_lift_len], target_len))
            emg_mean = emg_mean[:, 0:emg_lift_len]
            emg_std = emg_std[:, 0:emg_lift_len]
            emg_mean_out.append(resample_by_len(emg_mean[i, :], target_len))
            emg_std_out.append(resample_by_len(emg_std[i, :], target_len))
            arm_out1[i].append(arm1[i][1])
            arm_out2[i].append(arm2[i][1])
            arm_out3[i].append(arm3[i][1])
            # ml_out[i].append(ml[i][1])
            # tl_out[i].append(tl[i][1])
            # fa_out[i].append(fa[i][1])
        else:
            return os.error('State must include lift or down!')

    if include_state == 'lift and down':
        t_tor_out.append(np.concatenate([t_tor[0], t_tor[1]]))
        t_arm_out.append(np.concatenate([t_arm[0], t_arm[1]]))
        t_ang_out.append(np.concatenate([t_ang[0], t_ang[1]]))
        tor_out1.append(np.concatenate([tor1[0], tor1[1]]))
        tor_out2.append(np.concatenate([tor2[0], tor2[1]]))
        tor_out3.append(np.concatenate([tor3[0], tor3[1]]))
        ang_out1.append(np.concatenate([ang1[0], ang1[1]]))
        ang_out2.append(np.concatenate([ang2[0], ang2[1]]))
        ang_out3.append(np.concatenate([ang3[0], ang3[1]]))
    elif include_state == 'lift':
        t_tor_out.append(t_tor[0])
        t_arm_out.append(t_arm[0])
        t_ang_out.append(t_ang[0])
        tor_out1.append(tor1[0])
        tor_out2.append(tor2[0])
        tor_out3.append(tor3[0])
        ang_out1.append(ang1[0])
        ang_out2.append(ang2[0])
        ang_out3.append(ang3[0])
    elif include_state == 'down':
        t_tor_out.append(t_tor[1])
        t_arm_out.append(t_arm[1])
        t_ang_out.append(t_ang[1])
        tor_out1.append(tor1[1])
        tor_out2.append(tor2[1])
        tor_out3.append(tor3[1])
        ang_out1.append(ang1[1])
        ang_out2.append(ang2[1])
        ang_out3.append(ang3[1])

    if sport_label == 'deadlift':
        # if joint_idx == 'all':
        #     emg_rect_label = ['TA', 'GL', 'GM', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'ST', 'BF',
        #                       'GMax', 'GMed', 'PM', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
        # elif joint_idx == 'twist':
        #     emg_rect_label = ['PM', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
        emg_rect_label = musc_label
        emg_list = []
        for i in range(len(emg_rect_label)):
            if joint_idx == 'twist':
                [emg_rect, t1] = emg_rectification(emg[:, i + 14], fs, emg_rect_label[i], people, left)
            elif joint_idx == 'hip':
                if i == 0:
                    [emg_rect, t1] = emg_rectification(emg[:, i + 5], fs, emg_rect_label[i], people, left)
                else:
                    [emg_rect, t1] = emg_rectification(emg[:, i + 6], fs, emg_rect_label[i], people, left)
            elif joint_idx == 'knee':
                if i < 6:
                    [emg_rect, t1] = emg_rectification(emg[:, i + 1], fs, emg_rect_label[i], people, left)
                else:
                    [emg_rect, t1] = emg_rectification(emg[:, i + 3], fs, emg_rect_label[i], people, left)
            else:
                [emg_rect, t1] = emg_rectification(emg[:, i + 1], fs, emg_rect_label[i], people, left)
            emg_list.append(emg_rect)

    t1 = t1 - t_delta_emg + t_delta_joi
    if sport_label == 'deadlift':
        # if joint_idx == 'all':
        #     allk = [0] * 9 + [1, 2, 2, 21, 5, 1, 23, 49, 11, 39, 9]
        #     alli = [20] * 9 + [20, 21, 23, 25, 46, 51, 52, 75, 124, 135, 174]
        # elif joint_idx == 'twist' or joint_idx == 'hip' or joint_idx == 'knee':
        allk = related_muscle_num
        alli = related_muscle_idx

    if need_all_muscle is True:
        emg = [([]) for _ in range(len(muscle_idx))]
    else:
        emg = [([]) for _ in range(len(measured_muscle_idx))]
    time_emg = [[]]
    for t in t_tor_out[0]:
        idx = find_nearest_idx(t1, t)
        time_emg[0].append(t1[idx])
        if sport_label == 'bench_press' or sport_label == 'deadlift':
            for i in range(len(measured_muscle_idx)):
                emg[i].append(emg_list[i][idx])
            if need_all_muscle is True:
                for i in range(len(measured_muscle_idx)):
                    for k in range(allk[i]):
                        emg[alli[i] + k].append(emg_list[i][idx])

    emg_trend_u_out = np.asarray(emg_trend_u_out).squeeze()
    emg_trend_d_out = np.asarray(emg_trend_d_out).squeeze()
    emg_mean_out = np.asarray(emg_mean_out).squeeze()
    emg_std_out = np.asarray(emg_std_out).squeeze()
    arm_out1 = np.asarray(arm_out1).squeeze()
    arm_out2 = np.asarray(arm_out2).squeeze()
    arm_out3 = np.asarray(arm_out3).squeeze()
    # fa_out = np.asarray(fa_out).squeeze()
    tor_out1 = np.asarray(tor_out1).squeeze()
    tor_out2 = np.asarray(tor_out2).squeeze()
    tor_out3 = np.asarray(tor_out3).squeeze()
    ang_out1 = np.asarray(ang_out1).squeeze()
    ang_out2 = np.asarray(ang_out2).squeeze()
    ang_out3 = np.asarray(ang_out3).squeeze()
    t_tor_out = np.asarray(t_tor_out).squeeze()
    emg_mean = np.asarray(emg_mean).squeeze()
    emg_std = np.asarray(emg_std).squeeze()
    emg = np.asarray(emg).squeeze()
    time_emg = np.asarray(time_emg).squeeze()

    arm = np.dstack([arm_out1, arm_out2, arm_out3])
    torque = np.column_stack((tor_out1, tor_out2, tor_out3))

    output = {'emg_mean': emg_mean_out,
              'emg_std': emg_std_out,
              'arm': arm,
              'arm1': arm_out1,
              'arm2': arm_out2,
              'arm3': arm_out3,
              'torque': torque,
              'torque1': tor_out1,
              'torque2': tor_out2,
              'torque3': tor_out3,
              'angle1': ang_out1,
              'angle2': ang_out2,
              'angle3': ang_out3,
              'time': t_tor_out,
              'emg_mean_long': emg_mean,
              'emg_std_long': emg_std,
              'emg': emg,
              'time_emg': time_emg,
              'trend_u': emg_trend_u_out,
              'trend_d': emg_trend_d_out}
    return output


def read_groups_files(label='zhuo-right-3kg', idx=None):
    if idx is None:
        idx = ['1', '2', '3']
    if label == 'zhuo-right-3kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label, idx='2')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label, idx='3')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label, idx='5')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label, idx='6')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label, idx='8')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'zhuo-left-1kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label, idx='3')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label, idx='4')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label, idx='6')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label, idx='7')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label, idx='9')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3, emg_std4, emg_std5), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5), axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
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
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
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
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
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
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
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
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
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
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5),
                                       axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5),
                                      axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3, time_emg4, time_emg5), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3, trend_u4, trend_u5), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3, trend_d4, trend_d5), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-2.5kg_and_6.5kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='5')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='6')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='7')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='20')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='21')
        emg_mean6, emg_std6, arm6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6, trend_u6, trend_d6 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='22')
        emg_mean = np.concatenate(
            (emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5, emg_mean6), axis=1)
        emg_std = np.concatenate(
            (emg_std1, emg_std2, emg_std3, emg_std4, emg_std5, emg_std6), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5, arm6), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5, torque6),
                                axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5, time6), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5,
                                        emg_mean_long6), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5,
                                       emg_std_long6), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5, emg6), axis=1)
        time_emg = np.concatenate(
            (time_emg1, time_emg2, time_emg3, time_emg4, time_emg5, time_emg6), axis=0)
        trend_u = np.concatenate(
            (trend_u1, trend_u2, trend_u3, trend_u4, trend_u5, trend_u6), axis=0)
        trend_d = np.concatenate(
            (trend_d1, trend_d2, trend_d3, trend_d4, trend_d5, trend_d6), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-3kg_and_6.5kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='10')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='11')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='12')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='20')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='21')
        emg_mean6, emg_std6, arm6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6, trend_u6, trend_d6 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='22')

        emg_mean = np.concatenate(
            (emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5, emg_mean6), axis=1)
        emg_std = np.concatenate(
            (emg_std1, emg_std2, emg_std3, emg_std4, emg_std5, emg_std6), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5, arm6), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5, torque6),
                                axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5, time6), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5,
                                        emg_mean_long6), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5,
                                       emg_std_long6), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5, emg6), axis=1)
        time_emg = np.concatenate(
            (time_emg1, time_emg2, time_emg3, time_emg4, time_emg5, time_emg6), axis=0)
        trend_u = np.concatenate(
            (trend_u1, trend_u2, trend_u3, trend_u4, trend_u5, trend_u6), axis=0)
        trend_d = np.concatenate(
            (trend_d1, trend_d2, trend_d3, trend_d4, trend_d5, trend_d6), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-2.5kg_and_3kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='5')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='6')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='7')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='10')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='11')
        emg_mean6, emg_std6, arm6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6, trend_u6, trend_d6 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='12')
        emg_mean = np.concatenate(
            (emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5, emg_mean6), axis=1)
        emg_std = np.concatenate(
            (emg_std1, emg_std2, emg_std3, emg_std4, emg_std5, emg_std6), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5, arm6), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5, torque6),
                                axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5, time6), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5,
                                        emg_mean_long6), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5,
                                       emg_std_long6), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5, emg6), axis=1)
        time_emg = np.concatenate(
            (time_emg1, time_emg2, time_emg3, time_emg4, time_emg5, time_emg6), axis=0)
        trend_u = np.concatenate(
            (trend_u1, trend_u2, trend_u3, trend_u4, trend_u5, trend_u6), axis=0)
        trend_d = np.concatenate(
            (trend_d1, trend_d2, trend_d3, trend_d4, trend_d5, trend_d6), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-2.5kg_and_3kg_and_4kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='5')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='6')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='7')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='10')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='11')
        emg_mean6, emg_std6, arm6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6, trend_u6, trend_d6 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='12')
        emg_mean7, emg_std7, arm7, torque7, time7, emg_mean_long7, emg_std_long7, emg7, time_emg7, trend_u7, trend_d7 \
            = read_realted_files(label='chenzui-left-all-4kg', idx='15')
        emg_mean8, emg_std8, arm8, torque8, time8, emg_mean_long8, emg_std_long8, emg8, time_emg8, trend_u8, trend_d8 \
            = read_realted_files(label='chenzui-left-all-4kg', idx='16')
        emg_mean9, emg_std9, arm9, torque9, time9, emg_mean_long9, emg_std_long9, emg9, time_emg9, trend_u9, trend_d9 \
            = read_realted_files(label='chenzui-left-all-4kg', idx='17')

        emg_mean = np.concatenate(
            (emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5, emg_mean6, emg_mean7, emg_mean8, emg_mean9), axis=1)
        emg_std = np.concatenate(
            (emg_std1, emg_std2, emg_std3, emg_std4, emg_std5, emg_std6, emg_std7, emg_std8, emg_std9), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, arm9), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5, torque6, torque7, torque8, torque9),
                                axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5, time6, time7, time8, time9), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5,
                                        emg_mean_long6, emg_mean_long7, emg_mean_long8, emg_mean_long9), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5,
                                       emg_std_long6, emg_std_long7, emg_std_long8, emg_std_long9), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5, emg6, emg7, emg8, emg9), axis=1)
        time_emg = np.concatenate(
            (time_emg1, time_emg2, time_emg3, time_emg4, time_emg5, time_emg6, time_emg7, time_emg8, time_emg9), axis=0)
        trend_u = np.concatenate(
            (trend_u1, trend_u2, trend_u3, trend_u4, trend_u5, trend_u6, trend_u7, trend_u8, trend_u9), axis=0)
        trend_d = np.concatenate(
            (trend_d1, trend_d2, trend_d3, trend_d4, trend_d5, trend_d6, trend_d7, trend_d8, trend_d9), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'chenzui-left-2.5kg_and_3kg_and_6.5kg':
        emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='5')
        emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='6')
        emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files(label='chenzui-left-all-2.5kg', idx='7')
        emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, trend_u4, trend_d4 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='10')
        emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, trend_u5, trend_d5 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='11')
        emg_mean6, emg_std6, arm6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6, trend_u6, trend_d6 \
            = read_realted_files(label='chenzui-left-all-3kg', idx='12')
        emg_mean7, emg_std7, arm7, torque7, time7, emg_mean_long7, emg_std_long7, emg7, time_emg7, trend_u7, trend_d7 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='20')
        emg_mean8, emg_std8, arm8, torque8, time8, emg_mean_long8, emg_std_long8, emg8, time_emg8, trend_u8, trend_d8 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='21')
        emg_mean9, emg_std9, arm9, torque9, time9, emg_mean_long9, emg_std_long9, emg9, time_emg9, trend_u9, trend_d9 \
            = read_realted_files(label='chenzui-left-all-6.5kg', idx='22')

        emg_mean = np.concatenate(
            (emg_mean1, emg_mean2, emg_mean3, emg_mean4, emg_mean5, emg_mean6, emg_mean7, emg_mean8, emg_mean9), axis=1)
        emg_std = np.concatenate(
            (emg_std1, emg_std2, emg_std3, emg_std4, emg_std5, emg_std6, emg_std7, emg_std8, emg_std9), axis=1)
        arm = np.concatenate((arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, arm9), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque = np.concatenate((torque1, torque2, torque3, torque4, torque5, torque6, torque7, torque8, torque9),
                                axis=0)
        time = np.concatenate((time1, time2, time3, time4, time5, time6, time7, time8, time9), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3, emg_mean_long4, emg_mean_long5,
                                        emg_mean_long6, emg_mean_long7, emg_mean_long8, emg_mean_long9), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3, emg_std_long4, emg_std_long5,
                                       emg_std_long6, emg_std_long7, emg_std_long8, emg_std_long9), axis=1)
        emg = np.concatenate((emg1, emg2, emg3, emg4, emg5, emg6, emg7, emg8, emg9), axis=1)
        time_emg = np.concatenate(
            (time_emg1, time_emg2, time_emg3, time_emg4, time_emg5, time_emg6, time_emg7, time_emg8, time_emg9), axis=0)
        trend_u = np.concatenate(
            (trend_u1, trend_u2, trend_u3, trend_u4, trend_u5, trend_u6, trend_u7, trend_u8, trend_u9), axis=0)
        trend_d = np.concatenate(
            (trend_d1, trend_d2, trend_d3, trend_d4, trend_d5, trend_d6, trend_d7, trend_d8, trend_d9), axis=0)
        return emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'bp-chenzui-left-4kg' or label == 'bp-chenzui-left-5.5kg' or label == 'bp-chenzui-left-6.5kg' or label == 'bp-chenzui-left-7kg' or label == 'bp-chenzui-left-9.5kg':
        files1 = read_realted_files_bp(label, '1')
        files2 = read_realted_files_bp(label, '2')
        files3 = read_realted_files_bp(label, '3')
        files4 = read_realted_files_bp(label, '4')
        files5 = read_realted_files_bp(label, '5')
        files = [files1, files2, files3, files4, files5]
        output = {}
        axis0 = ['torque1', 'torque2', 'time', 'time_emg', 'angle1', 'angle2']
        for j in files1:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'bp-chenzui-left-mix':
        files1 = read_realted_files_bp('bp-chenzui-left-4kg', '1')
        files2 = read_realted_files_bp('bp-chenzui-left-4kg', '2')
        files3 = read_realted_files_bp('bp-chenzui-left-4kg', '3')
        files4 = read_realted_files_bp('bp-chenzui-left-5.5kg', '1')
        files5 = read_realted_files_bp('bp-chenzui-left-5.5kg', '2')
        files6 = read_realted_files_bp('bp-chenzui-left-5.5kg', '3')
        files7 = read_realted_files_bp('bp-chenzui-left-7kg', '1')
        files8 = read_realted_files_bp('bp-chenzui-left-7kg', '2')
        files9 = read_realted_files_bp('bp-chenzui-left-7kg', '3')
        files = [files1, files2, files3, files4, files5, files6, files7, files8, files9]
        output = {}
        axis0 = ['torque1', 'torque2', 'time', 'time_emg', 'angle1', 'angle2']
        for j in files1:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'bp-chenzui-left-9.5kg-before':
        files1 = read_realted_files_bp(label, '1')
        files2 = read_realted_files_bp(label, '2')
        files3 = read_realted_files_bp(label, '3')
        files4 = read_realted_files_bp(label, '4')
        files5 = read_realted_files_bp(label, '5')
        files = [files1, files2, files3, files4, files5]
        output = {}
        axis0 = ['torque1', 'torque2', 'time', 'time_emg', 'angle1', 'angle2']
        for j in files1:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'bp-zhuo-right-3kg':
        emg_mean1, emg_std1, arm11, arm21, torque11, torque21, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files_bp(label, '7')
        emg_mean2, emg_std2, arm12, arm22, torque12, torque22, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files_bp(label, '8')
        emg_mean3, emg_std3, arm13, arm23, torque13, torque23, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files_bp(label, '9')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3), axis=1)
        arm1 = np.concatenate((arm11, arm12, arm13), axis=1)
        arm2 = np.concatenate((arm21, arm22, arm23), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque1 = np.concatenate((torque11, torque12, torque13), axis=0)
        torque2 = np.concatenate((torque21, torque22, torque23), axis=0)
        time = np.concatenate((time1, time2, time3), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3), axis=1)
        emg = np.concatenate((emg1, emg2, emg3), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3), axis=0)
        return emg_mean, emg_std, arm1, arm2, torque1, torque2, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'bp-zhuo-right-4kg':
        emg_mean1, emg_std1, arm11, arm21, torque11, torque21, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, trend_u1, trend_d1 \
            = read_realted_files_bp(label, '1')
        emg_mean2, emg_std2, arm12, arm22, torque12, torque22, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, trend_u2, trend_d2 \
            = read_realted_files_bp(label, '2')
        emg_mean3, emg_std3, arm13, arm23, torque13, torque23, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, trend_u3, trend_d3 \
            = read_realted_files_bp(label, '3')

        emg_mean = np.concatenate((emg_mean1, emg_mean2, emg_mean3), axis=1)
        emg_std = np.concatenate((emg_std1, emg_std2, emg_std3), axis=1)
        arm1 = np.concatenate((arm11, arm12, arm13), axis=1)
        arm2 = np.concatenate((arm21, arm22, arm23), axis=1)
        # fa = np.concatenate((fa1, fa2, fa3, fa4, fa5, fa6), axis=1)
        torque1 = np.concatenate((torque11, torque12, torque13), axis=0)
        torque2 = np.concatenate((torque21, torque22, torque23), axis=0)
        time = np.concatenate((time1, time2, time3), axis=0)
        emg_mean_long = np.concatenate((emg_mean_long1, emg_mean_long2, emg_mean_long3), axis=1)
        emg_std_long = np.concatenate((emg_std_long1, emg_std_long2, emg_std_long3), axis=1)
        emg = np.concatenate((emg1, emg2, emg3), axis=1)
        time_emg = np.concatenate((time_emg1, time_emg2, time_emg3), axis=0)
        trend_u = np.concatenate((trend_u1, trend_u2, trend_u3), axis=0)
        trend_d = np.concatenate((trend_d1, trend_d2, trend_d3), axis=0)
        return emg_mean, emg_std, arm1, arm2, torque1, torque2, time, emg_mean_long, emg_std_long, emg, time_emg, trend_u, trend_d
    elif label == 'bp-chenzui-left-4.1kg':
        label = 'bp-chenzui-left-4kg'
        files1 = read_realted_files_bp(label, '1')
        files2 = read_realted_files_bp(label, '2')
        files3 = read_realted_files_bp(label, '3')
        files4 = read_realted_files_bp(label, '4')
        files5 = read_realted_files_bp(label, '5')
        files = [files1, files2, files3, files4, files5]
        output = {}
        axis0 = ['torque1', 'torque2', 'time', 'time_emg', 'angle1', 'angle2']
        for j in files1:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    # elif label == 'bp-yuetian-right-20kg':
    #     files1 = read_realted_files_bp(label, '1')
    #     files2 = read_realted_files_bp(label, '2')
    #     files3 = read_realted_files_bp(label, '3')
    #     files4 = read_realted_files_bp(label, '4')
    #     files5 = read_realted_files_bp(label, '5')
    #     files = [files1, files2, files3, files4, files5]
    #     output = {}
    #     axis0 = ['torque1', 'torque2', 'time', 'time_emg', 'angle1', 'angle2']
    #     for j in files1:
    #         # print(j)
    #         if j in axis0:
    #             idx_axis = 0
    #         else:
    #             idx_axis = 1
    #         output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
    #     return output
    # elif label == 'bp-yuetian-right-mix':
    #     files1 = read_realted_files_bp('bp-yuetian-right-20kg', '1')
    #     files2 = read_realted_files_bp('bp-yuetian-right-20kg', '2')
    #     files3 = read_realted_files_bp('bp-yuetian-right-20kg', '3')
    #     files4 = read_realted_files_bp('bp-yuetian-right-30kg', '1')
    #     files5 = read_realted_files_bp('bp-yuetian-right-30kg', '2')
    #     files6 = read_realted_files_bp('bp-yuetian-right-30kg', '3')
    #     files7 = read_realted_files_bp('bp-yuetian-right-50kg', '1')
    #     files8 = read_realted_files_bp('bp-yuetian-right-50kg', '2')
    #     files9 = read_realted_files_bp('bp-yuetian-right-50kg', '3')
    #     files = [files1, files2, files3, files4, files5, files6, files7, files8, files9]
    #     output = {}
    #     axis0 = ['torque1', 'torque2', 'time', 'time_emg', 'angle1', 'angle2']
    #     for j in files1:
    #         # print(j)
    #         if j in axis0:
    #             idx_axis = 0
    #         else:
    #             idx_axis = 1
    #         output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
    #     return output
    elif label == 'bp-yuetian-right-20kg' or label == 'bp-yuetian-right-30kg' or label == 'bp-yuetian-right-40kg' or label == 'bp-yuetian-right-50kg' or label == 'bp-yuetian-right-60kg':
        all_possible_idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert all(elem in all_possible_idx for elem in idx)
        axis0 = ['torque1', 'torque2', 'torque3', 'torque', 'time', 'time_emg', 'angle1', 'angle2', 'angle3']
        files = []
        output = {}
        for i in idx:
            file = read_realted_files_bp(label, i)
            files.append(file)
        for j in files[0]:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'bp-yuetian-right-all':
        labels = ['bp-yuetian-right-20kg', 'bp-yuetian-right-40kg', 'bp-yuetian-right-60kg']
        idxs = [['2', '3', '4', '5', '6'],
                ['2', '3', '4', '5', '7'],
                ['2', '3', '4', '5']]
        axis0 = ['torque1', 'torque2', 'torque3', 'torque', 'time', 'time_emg', 'angle1', 'angle2', 'angle3']
        files = []
        output = {}
        for k in range(len(labels)):
            label = labels[k]
            idx = idxs[k]
            for i in idx:
                file = read_realted_files_bp(label, i)
                files.append(file)
        for j in files[0]:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'dl-yuetian-right-35kg' or label == 'dl-yuetian-right-45kg' or label == 'dl-yuetian-right-65kg' or label == 'dl-yuetian-right-75kg':
        all_possible_idx = ['1', '2', '3', '4', '5', '6']
        assert all(elem in all_possible_idx for elem in idx)
        axis0 = ['torque1', 'torque2', 'torque3', 'torque', 'time', 'time_emg', 'angle1', 'angle2', 'angle3']
        files = []
        output = {}
        for i in idx:
            file = read_realted_files_dl(label, i)
            files.append(file)
        for j in files[0]:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'dl-yuetian-right-all':
        labels = ['dl-yuetian-right-35kg', 'dl-yuetian-right-45kg', 'dl-yuetian-right-75kg']
        idxs = [['2', '3', '5', '6'],
                ['1', '2', '3', '5'],
                ['1', '2', '3', '4', '5', '6']]
        axis0 = ['torque1', 'torque2', 'torque3', 'torque', 'time', 'time_emg', 'angle1', 'angle2', 'angle3']
        files = []
        output = {}
        for k in range(len(labels)):
            label = labels[k]
            idx = idxs[k]
            for i in idx:
                file = read_realted_files_dl(label, i)
                files.append(file)
        for j in files[0]:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'bp-kehan-right-20kg' or label == 'bp-kehan-right-30kg' or label == 'bp-kehan-right-40kg' or label == 'bp-kehan-right-50kg' or label == 'bp-kehan-right-45kg':
        all_possible_idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        assert all(elem in all_possible_idx for elem in idx)
        axis0 = ['torque1', 'torque2', 'torque3', 'torque', 'time', 'time_emg', 'angle1', 'angle2', 'angle3']
        files = []
        output = {}
        for i in idx:
            file = read_realted_files_bp(label, i)
            files.append(file)
        for j in files[0]:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    elif label == 'bp-kehan-right-all':
        labels = ['bp-kehan-right-30kg', 'bp-kehan-right-40kg', 'bp-kehan-right-50kg']
        idxs = [['1', '3', '4', '5', '6'],
                ['1', '2', '3', '4', '6'],
                ['2', '3']]
        axis0 = ['torque1', 'torque2', 'torque3', 'torque', 'time', 'time_emg', 'angle1', 'angle2', 'angle3']
        files = []
        output = {}
        for k in range(len(labels)):
            label = labels[k]
            idx = idxs[k]
            for i in idx:
                file = read_realted_files_bp(label, i)
                files.append(file)
        for j in files[0]:
            # print(j)
            if j in axis0:
                idx_axis = 0
            else:
                idx_axis = 1
            output.update({j: np.concatenate([files[i][j] for i in range(len(files))], axis=idx_axis)})
        return output
    else:
        print('No such label!')


def read_mat_files(label):
    # fvfl = [[], [], []]
    # acti = [[], [], []]
    # marm = [[], [], []]
    fvfl = [[], [], [], [], [], [], []]
    acti = [[], [], [], [], [], [], []]
    marm = [[], [], [], [], [], [], []]
    m_idx = ['BICsht', 'BRA', 'BRD', 'BIClon', 'TRIlon', 'TRIlat', 'TRImed']
    if label == '2kg':
        data = pd.read_excel('files/yuetian/2kg_division3_new.xlsx')
    elif label == '3.5kg':
        data = pd.read_excel('files/yuetian/3.5kg_division3_new.xlsx')
    elif label == '5kg':
        data = pd.read_excel('files/yuetian/5kg_division3_new.xlsx')

    else:
        return 0
    slicing_num = 8
    time = data['time'][::slicing_num]
    torque = data['torque'][::slicing_num]
    for i in range(len(m_idx)):
        fvfl[i] = data[m_idx[i] + '_Multipliers'][::slicing_num]
        acti[i] = data[m_idx[i] + '_Activation'][::slicing_num]
        marm[i] = data[m_idx[i] + '_MmtArm'][::slicing_num]
        marm[i] = np.ones_like(marm[i]) * np.mean(marm[i])

    time = np.asarray(time)
    torque = np.asarray(torque)
    fvfl = np.asarray(fvfl)
    acti = np.asarray(acti)
    marm = np.asarray(marm)
    return time, torque, fvfl, acti, marm


if __name__ == '__main__':
    # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    #     = read_groups_files('chenzui-left-3kg')
    # read_mat_files('2kg')
    output = read_groups_files('bp-chenzui-left-4.1kg')
