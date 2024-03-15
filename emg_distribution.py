import matplotlib.pyplot as plt
import pandas as pd

from require import *
from basic import *


def emg_file_progressing(emg, fs=1000, people=None, sport='biceps_curl'):
    if sport == 'biceps_curl':
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
    elif sport == 'bench_press':
        [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people)
        [emg_TRI, t2] = emg_rectification(emg[:, 2], fs, 'TRI', people)
        [emg_ANT, t3] = emg_rectification(emg[:, 3], fs, 'ANT', people)
        [emg_POS, t4] = emg_rectification(emg[:, 4], fs, 'POS', people)
        [emg_PEC, t5] = emg_rectification(emg[:, 5], fs, 'PEC', people)
        [emg_LAT, t6] = emg_rectification(emg[:, 6], fs, 'LAT', people)
        emg_list = np.asarray([emg_BIC, emg_TRI, emg_ANT, emg_POS, emg_PEC, emg_LAT])
        t_list = np.asarray([t1, t2, t3, t4, t5, t6])
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


def calculate_chenzui_emg_distribution(label='3kg'):
    unified_len = 1000
    fs = 2000
    num = 0
    # sport_label = 'biceps_curl'
    if label == '3kg':
        file_folder = 'files/chenzui-3kg/'
        files = [file_folder + 'emg-11.npy',
                 file_folder + 'emg-13.npy',
                 file_folder + 'emg-14.npy',
                 file_folder + 'emg-15.npy',
                 file_folder + 'emg-16.npy',
                 file_folder + 'emg-17.npy',
                 file_folder + 'emg-18.npy']
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
        file_folder = 'files/chenzui-5.5kg/'
        files = [file_folder + '7.npy',
                 file_folder + '3.npy',
                 file_folder + '4.npy',
                 file_folder + '5.npy',
                 file_folder + '6.npy',
                 file_folder + '8.npy',
                 file_folder + '9.npy']
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
        file_folder = 'files/chenzui-all/'
        files = [file_folder + '2.5-5.npy',
                 file_folder + '2.5-6.npy',
                 file_folder + '2.5-7.npy',
                 file_folder + '2.5-8.npy',
                 file_folder + '2.5-9.npy']
        t_delta_emg = [3.464, 3.7925, 3.2645, 3.178, 4.6665]
        t_delta_joi = [9.733, 7.65, 8.583, 11.249, 9.466]
        timestep_emg = [[16.416, 27.065, 27.065, 41.681],
                        [13.750, 24.782, 24.782, 42.731],
                        [13.649, 25.982, 25.982, 41.598],
                        [15.832, 28.482, 28.482, 44.864],
                        [15.549, 26.365, 26.599, 40.031]]
    elif label == '3kg-all':
        file_folder = 'files/chenzui-all/'
        files = [file_folder + '3-10.npy',
                 file_folder + '3-11.npy',
                 file_folder + '3-12.npy',
                 file_folder + '3-13.npy',
                 file_folder + '3-14.npy']
        t_delta_emg = [7.177, 3.5745, 5.055, 4.588, 9.098]
        t_delta_joi = [10.1, 8.866, 9.283, 6.983, 12.816]
        timestep_emg = [[15.599, 26.249, 26.532, 39.348],
                        [15.216, 25.582, 25.582, 40.781],
                        [16.066, 28.299, 28.299, 40.882],
                        [12.049, 22.699, 22.699, 34.965],
                        [17.849, 29.365, 29.365, 43.381]]
    elif label == '4kg-all':
        file_folder = 'files/chenzui-all/'
        files = [file_folder + '4-15.npy',
                 file_folder + '4-16.npy',
                 file_folder + '4-17.npy']
        t_delta_emg = [3.3555, 4.450, 3.735]
        t_delta_joi = [9.7, 8.783, 7.233]
        timestep_emg = [[14.966, 24.216, 24.216, 37.399],
                        [13.133, 23.899, 23.899, 37.565],
                        [11.433, 22.516, 22.516, 35.682]]
    elif label == '6.5kg-all':
        file_folder = 'files/chenzui-all/'
        files = [file_folder + '6.5-18.npy',
                 file_folder + '6.5-19.npy',
                 file_folder + '6.5-20.npy',
                 file_folder + '6.5-21.npy',
                 file_folder + '6.5-22.npy']
        t_delta_emg = [4.869, 4.0085, 6.505, 3.8335, 13.803]
        t_delta_joi = [6.183, 6.033, 8.0, 6.65, 15.4]
        timestep_emg = [[11.233, 23.465, 23.465, 37.948],
                        [11.833, 25.132, 25.132, 36.548],
                        [14.633, 27.399, 27.399, 40.465],
                        [11.9, 24.599, 24.599, 37.465],
                        [19.466, 30.415, 31.749, 45.781]]  # 6.5kg
    # elif label == '6.5kg-cts':
    #     file_folder = 'files/chenzui-all/'
    #     files = [file_folder+'6.5.npy',
    #              file_folder+'6.5.npy',
    #              file_folder+'6.5.npy',
    #              file_folder+'6.5.npy',
    #              file_folder+'6.5.npy']
    #     t_delta_emg = [5.6085, 5.6085, 5.6085, 5.6085, 5.6085]
    #     t_delta_joi = [12.617, 12.617, 12.617, 12.617, 12.617]
    #     timestep_emg = [[18.016, 28.232, 28.232, 38.849],
    #                     [47.148, 59.531, 59.531, 70.264],
    #                     [78.747, 92.862, 92.862, 104.112],
    #                     [112.511, 123.794, 123.794, 133.594],
    #                     [141.76, 153.243, 153.243, 162.176]]  # 6.5kg
    elif label == '4kg-cts':
        file_folder = 'files/chenzui-cts/'
        files = [file_folder + '5-4.npy'] * 16
        t_delta_emg = [83.755] * 16
        t_delta_joi = [90.078] * 16
        timestep_emg = [[94.278, 101.394, 102.828, 110.361],
                        [111.81, 119.16, 120.043, 128.026],
                        [129.809, 136.259, 137.209, 144.775],
                        [146.592, 153.675, 154.458, 161.574],
                        [163.774, 170.157, 171.207, 178.307],
                        [181.373, 187.506, 188.44, 194.589],
                        [197.923, 203.989, 204.689, 210.789],
                        [213.938, 219.421, 220.321, 225.204],
                        [229.888, 235.954, 236.854, 243.054],
                        [246.603, 252.453, 253.42, 259.369],
                        [263.802, 270.019, 270.835, 276.568],
                        [280.901, 286.851, 287.751, 293.584],
                        [298.084, 304.317, 305.25, 310.983],
                        [315.883, 322.199, 322.999, 328.382],
                        [331.932, 337.632, 338.582, 343.548],
                        [348.481, 353.631, 354.631, 359.481]]  # 4kg
    elif label == '5.5kg-cts':
        file_folder = 'files/chenzui-cts/'
        files = [file_folder + '4-5.5.npy'] * 9
        t_delta_emg = [3.6855] * 9
        t_delta_joi = [10.216] * 9
        timestep_emg = [[15.015, 23.482, 23.881, 32.714],
                        [34.464, 44.23, 44.464, 53.663],
                        [56.463, 65.763, 66.696, 75.962],
                        [78.612, 86.495, 86.962, 94.645],
                        [99.078, 106.594, 107.344, 114.027],
                        [118.827, 125.426, 125.76, 131.693],
                        [137.409, 144.575, 144.642, 150.908],
                        [155.875, 161.241, 161.491, 166.324],
                        [171.507, 175.94, 176.357, 181.09]]  # 5.5kg
    elif label == '6.5kg-cts':
        file_folder = 'files/chenzui-cts/'
        files = [file_folder + '2-6.5.npy'] * 10
        t_delta_emg = [4.2] * 10
        t_delta_joi = [10.717] * 10
        timestep_emg = [[14.966, 21.699, 25.899, 35.565],
                        [36.365, 43.598, 47.098, 56.881],
                        [57.98, 64.464, 68.996, 77.879],
                        [79.33, 85.529, 89.429, 97.562],
                        [99.412, 106.145, 109.745, 119.378],
                        [122.227, 127.644, 132.444, 139.76],
                        [143.56, 148.509, 152.726, 159.975],
                        [164.809, 171.175, 173.458, 180.291],
                        [184.341, 190.107, 191.307, 197.44],
                        [201.807, 206.856, 207.889, 213.623]]  # 6.5kg
    elif label == 'all-1':
        file_folder = 'files/chenzui-all/'
        files = [file_folder + '2.5-5.npy',
                 file_folder + '2.5-6.npy',
                 file_folder + '2.5-7.npy',
                 file_folder + '2.5-8.npy',
                 file_folder + '2.5-9.npy',
                 file_folder + '3-10.npy',
                 file_folder + '3-11.npy',
                 file_folder + '3-12.npy',
                 file_folder + '3-13.npy',
                 file_folder + '3-14.npy',
                 file_folder + '4-15.npy',
                 file_folder + '4-16.npy',
                 file_folder + '4-17.npy',
                 file_folder + '6.5-18.npy',
                 file_folder + '6.5-19.npy',
                 file_folder + '6.5-20.npy',
                 file_folder + '6.5-21.npy',
                 file_folder + '6.5-22.npy']
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
                        [15.549, 26.365, 26.599, 40.031],  # 2.5kg
                        [15.599, 26.249, 26.532, 39.348],
                        [15.216, 25.582, 25.582, 40.781],
                        [16.066, 28.299, 28.299, 40.882],
                        [12.049, 22.699, 22.699, 34.965],
                        [17.849, 29.365, 29.365, 43.381],  # 3kg
                        [14.966, 24.216, 24.216, 37.399],
                        [13.133, 23.899, 23.899, 37.565],
                        [11.433, 22.516, 22.516, 35.682],  # 4kg
                        [11.233, 23.465, 23.465, 37.948],
                        [11.833, 25.132, 25.132, 36.548],
                        [14.633, 27.399, 27.399, 40.465],
                        [11.9, 24.599, 24.599, 37.465],
                        [19.466, 30.415, 31.749, 45.781]]  # 6.5kg
    elif label == 'all':
        file_folder = 'files/chenzui-all/'
        files = [file_folder + '2.5-5.npy',
                 file_folder + '/2.5-6.npy',
                 file_folder + '2.5-7.npy',
                 file_folder + '2.5-8.npy',
                 file_folder + '2.5-9.npy',
                 file_folder + '3-10.npy',
                 file_folder + '3-11.npy',
                 file_folder + '3-12.npy',
                 file_folder + '3-13.npy',
                 file_folder + '3-14.npy',
                 file_folder + '4-15.npy',
                 file_folder + '4-16.npy',
                 file_folder + '4-17.npy']
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
                        [15.549, 26.365, 26.599, 40.031],  # 2.5kg
                        [15.599, 26.249, 26.532, 39.348],
                        [15.216, 25.582, 25.582, 40.781],
                        [16.066, 28.299, 28.299, 40.882],
                        [12.049, 22.699, 22.699, 34.965],
                        [17.849, 29.365, 29.365, 43.381],  # 3kg
                        [14.966, 24.216, 24.216, 37.399],
                        [13.133, 23.899, 23.899, 37.565],
                        [11.433, 22.516, 22.516, 35.682]]  # 4kg
    elif label == 'bp-4kg':
        file_folder = 'files/bench press/chenzui-4kg/'
        files = [file_folder + '6.npy'] * 3
        sport_label = 'bench_press'
        t_delta_emg = [7.9805] * 3
        t_delta_joi = [9.199] * 3
        timestep_emg = [[14.899, 22.932, 22.932, 27.865],
                        [32.565, 41.198, 41.198, 47.631],
                        [52.364, 62.597, 62.597, 68.196]]  # 4kg
    else:
        print('No label:', label)

    data_set_number = len(files)
    emg_all = [([]) for _ in range(data_set_number)]
    t_emg_all = [([]) for _ in range(data_set_number)]
    for i in range(data_set_number):
        emg_all[i] = np.load(files[i])
        [emg_all[i], t_emg_all[i]] = emg_file_progressing(emg_all[i], fs, 'chenzui', sport_label)
        t_emg_all[i] = t_emg_all[i] - t_delta_emg[i] + t_delta_joi[i]

    if sport_label == 'biceps_curl':
        if include_TRI is True:
            muscles = [([], [], [], []) for _ in range(data_set_number)]  # number of muscle
        else:
            muscles = [([], [], []) for _ in range(data_set_number)]  # number of muscle
    if sport_label == 'bench_press':
        muscles = [([], [], [], [], [], []) for _ in range(data_set_number)]  # number of muscle
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
        data_trend_u[i] = np.asarray(
            [(np.max(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray(
            [(np.min(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
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

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    musc_label = ['Biceps', 'Triceps', 'Anterior', 'Posterior', 'Pectoralis', 'Latissimus']
    if sport_label == 'biceps_curl':
        plt.figure(figsize=(6, 6.7))
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
    elif sport_label == 'bench_press':
        plt.figure(figsize=(6, 7.7))
        for j in range(len(muscle_idx)):
            plt.subplot(len(muscle_idx), 1, j + 1)
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

    if sport_label == 'biceps_curl':
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
            num = num + 1
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
    elif sport_label == 'bench_press':
        plt.figure(figsize=(6, 7.7))
        for j in range(len(muscle_idx)):
            plt.subplot(len(muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

    if sport_label == 'biceps_curl':
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
    elif sport_label == 'bench_press':
        plt.figure(figsize=(6, 7.7))
        for j in range(len(muscle_idx)):
            plt.subplot(len(muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j], color='papayawhip')
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

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
    elif label == '6.5kg-cts':
        np.save('emg/chenzui_mean_6.5kg_cts', data_mean)
        np.save('emg/chenzui_std_6.5kg_cts', data_std)
        np.save('emg/chenzui_trend_u_6.5kg_cts', data_trend_u)
        np.save('emg/chenzui_trend_d_6.5kg_cts', data_trend_d)
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
    elif label == '4kg-cts':
        np.save('emg/chenzui_mean_4kg_cts', data_mean)
        np.save('emg/chenzui_std_4kg_cts', data_std)
        np.save('emg/chenzui_trend_u_4kg_cts', data_trend_u)
        np.save('emg/chenzui_trend_d_4kg_cts', data_trend_d)
    elif label == '5.5kg-cts':
        np.save('emg/chenzui_mean_5.5kg_cts', data_mean)
        np.save('emg/chenzui_std_5.5kg_cts', data_std)
        np.save('emg/chenzui_trend_u_5.5kg_cts', data_trend_u)
        np.save('emg/chenzui_trend_d_5.5kg_cts', data_trend_d)
    elif label == '6.5kg-cts':
        np.save('emg/chenzui_mean_6.5kg_cts', data_mean)
        np.save('emg/chenzui_std_6.5kg_cts', data_std)
        np.save('emg/chenzui_trend_u_6.5kg_cts', data_trend_u)
        np.save('emg/chenzui_trend_d_6.5kg_cts', data_trend_d)
    elif label == 'all' or label == 'all-1':
        np.save('emg/chenzui_mean_all', data_mean)
        np.save('emg/chenzui_std_all', data_std)
        np.save('emg/chenzui_trend_u_all', data_trend_u)
        np.save('emg/chenzui_trend_d_all', data_trend_d)
    elif label == 'bp-4kg':
        np.save('emg/bp-chenzui_mean_4kg', data_mean)
        np.save('emg/bp-chenzui_std_4kg', data_std)
        np.save('emg/bp-chenzui_trend_u_4kg', data_trend_u)
        np.save('emg/bp-chenzui_trend_d_4kg', data_trend_d)

    # print(np.max(yt[:, 0, :, :]))
    # print(np.max(yt[:, 1, :, :]))
    # print(np.max(yt[:, 2, :, :]))
    # print(np.max(yt[:, 3, :, :]))
    # print(np.max(yt[:, 4, :, :]))
    # plt.show()


def calculate_lizhuo_emg_distribution(label='1kg'):
    unified_len = 1000
    fs = 2000
    num = 0
    if label == '1kg':
        file_folder = 'files/lizhuo-1kg/'
        files = [file_folder + '2.npy',
                 file_folder + '3.npy',
                 file_folder + '4.npy',
                 file_folder + '6.npy',
                 file_folder + '7.npy',
                 file_folder + '8.npy',
                 file_folder + '9.npy']
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
        file_folder = 'files/lizhuo-3kg/'
        files = [file_folder + '1.npy',
                 file_folder + '2.npy',
                 file_folder + '3.npy',
                 file_folder + '5.npy',
                 file_folder + '6.npy',
                 # file_folder+'7.npy',
                 file_folder + '8.npy',
                 file_folder + '9.npy',
                 file_folder + '10.npy',
                 file_folder + '11.npy',
                 file_folder + '12.npy']
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
        file_folder = 'files/lizhuo-3kg/'
        files = [file_folder + '12.npy',
                 file_folder + '12.npy',
                 file_folder + '12.npy']
        t_delta_emg = [7.29, 7.29, 7.29]
        t_delta_joi = [14.749, 14.749, 14.749]
        timestep_emg = [[19.032, 21.649, 21.649, 24.465],  # 2.62s, 2.82s
                        [24.465, 26.865, 26.865, 30.165],  # 2.40s, 3.30s
                        [30.665, 33.198, 33.198, 36.498]]  # 2.53s, 3.30s
    elif label == 'bp-3kg':
        file_folder = 'files/bench press/lizhuo/'
        files = [file_folder + '3.npy'] * 9
        # sport_label = 'bench_press'
        t_delta_emg = [7.61] * 9
        t_delta_joi = [12.699] * 9
        timestep_emg = [[34.131, 36.231, 36.231, 41.498],
                        [43.364, 44.364, 44.364, 47.197],
                        [56.997, 58.297, 58.297, 62.063],
                        [63.330, 64.430, 64.430, 69.663],
                        [70.963, 72.763, 73.429, 77.162],
                        [78.629, 80.062, 80.062, 84.329],
                        [85.929, 87.495, 87.495, 92.628],
                        [104.361, 106.627, 106.627, 111.46],
                        [114.094, 116.527, 116.893, 120.86]]  # 3kg
    elif label == 'bp-4kg':
        file_folder = 'files/bench press/lizhuo/'
        files = [file_folder + '4.npy'] * 9
        # sport_label = 'bench_press'
        t_delta_emg = [4.749] * 9
        t_delta_joi = [12.9] * 9
        timestep_emg = [[26.132, 28.999, 28.999, 33.232],
                      [35.032, 37.165, 37.165, 41.765],
                      [43.631, 46.098, 46.098, 49.365],
                      [51.698, 53.731, 53.731, 57.464],
                      [59.697, 61.264, 61.264, 63.93],
                      [66.497, 68.063, 68.063, 70.93],
                      [73.263, 74.663, 74.663, 77.23],
                      [81.03, 82.263, 82.263, 85.096],
                      [87.996, 89.696, 89.696, 92.929],
                      [95.729, 97.129, 97.129, 100.628]]  # 3kg
    else:
        print('No label:', label)

    data_set_number = len(files)
    emg_all = [([]) for _ in range(data_set_number)]
    t_emg_all = [([]) for _ in range(data_set_number)]
    for i in range(data_set_number):
        emg_all[i] = np.load(files[i])
        if seven_six is True:
            emg_all[i][:, 6] = emg_all[i][:, 5]
        [emg_all[i], t_emg_all[i]] = emg_file_progressing(emg_all[i], fs, 'zhuo', sport_label)
        t_emg_all[i] = t_emg_all[i] - t_delta_emg[i] + t_delta_joi[i]

    if sport_label == 'biceps_curl':
        if include_TRI is True:
            muscles = [([], [], [], []) for _ in range(data_set_number)]  # number of muscle
        else:
            muscles = [([], [], []) for _ in range(data_set_number)]  # number of muscle
    if sport_label == 'bench_press':
        muscles = [([], [], [], [], [], []) for _ in range(data_set_number)]  # number of muscle
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
    # Fs = 200
    # Wn = LOWPASSRATE / (Fs / 2)
    # [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    # data = scipy.signal.filtfilt(b, a, data, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波
    #
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
        data_trend_u[i] = np.asarray(
            [(np.max(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray(
            [(np.min(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        # data_trend_u[i] = np.asarray([np.max(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])
        # data_trend_d[i] = np.asarray([np.min(data[i, :, j] / data[i, :, 0]) for j in range(data.shape[2] - 1)])
        # if muscle_LAT is True and i == len(muscle_idx) - 1:
        #     data_mean[i] = np.asarray([np.mean(data[i - 1, :, j]) for j in range(data.shape[2])])
        #     data_std[i] = np.asarray([np.std(data[i - 1, :, j]) for j in range(data.shape[2])])
        #     data_trend_u[i] = np.asarray(
        #         [(np.max(data[i - 1, :, j + 1] - data[i - 1, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        #     data_trend_d[i] = np.asarray(
        #         [(np.min(data[i - 1, :, j + 1] - data[i - 1, :, j]) / 0.01) for j in range(data.shape[2] - 1)])

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    musc_label = ['Biceps', 'Triceps', 'Anterior', 'Posterior', 'Pectoralis', 'Latissimus']
    if sport_label == 'biceps_curl':
        plt.figure(figsize=(6, 6.7))
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
    elif sport_label == 'bench_press':
        plt.figure(figsize=(6, 7.7))
        for j in range(len(muscle_idx)):
            plt.subplot(len(muscle_idx), 1, j + 1)
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

    if sport_label == 'biceps_curl':
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
            num = num + 1
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
    elif sport_label == 'bench_press':
        plt.figure(figsize=(6, 7.7))
        for j in range(len(muscle_idx)):
            plt.subplot(len(muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

    if sport_label == 'biceps_curl':
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
    elif sport_label == 'bench_press':
        plt.figure(figsize=(6, 7.7))
        for j in range(len(muscle_idx)):
            plt.subplot(len(muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j], color='papayawhip')
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

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
    elif label == 'bp-3kg':
        np.save('emg/bp-lizhuo_mean_3kg', data_mean)
        np.save('emg/bp-lizhuo_std_3kg', data_std)
        np.save('emg/bp-lizhuo_trend_u_3kg', data_trend_u)
        np.save('emg/bp-lizhuo_trend_d_3kg', data_trend_d)
    elif label == 'bp-4kg':
        np.save('emg/bp-lizhuo_mean_4kg', data_mean)
        np.save('emg/bp-lizhuo_std_4kg', data_std)
        np.save('emg/bp-lizhuo_trend_u_4kg', data_trend_u)
        np.save('emg/bp-lizhuo_trend_d_4kg', data_trend_d)

    # print(np.max(yt[:, 0, :, :]))
    # print(np.max(yt[:, 1, :, :]))
    # print(np.max(yt[:, 2, :, :]))
    # print(np.max(yt[:, 3, :, :]))
    # print(np.max(yt[:, 4, :, :]))
    # plt.show()


if __name__ == '__main__':
    calculate_chenzui_emg_distribution(label='bp-4kg')
    # calculate_lizhuo_emg_distribution(label='1kg')
    plt.show()
