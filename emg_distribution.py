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
        if date == '240408':
            [emg_BIC, t1] = emg_rectification(emg[:, 1], fs, 'BIC', people, left)
            [emg_TRI, t2] = emg_rectification(emg[:, 2], fs, 'TRI', people, left)
            [emg_ANT, t3] = emg_rectification(emg[:, 3], fs, 'ANT', people, left)
            [emg_POS, t4] = emg_rectification(emg[:, 4], fs, 'POS', people, left)
            [emg_PEC, t5] = emg_rectification(emg[:, 5], fs, 'PEC', people, left)
            [emg_LAT, t6] = emg_rectification(emg[:, 6], fs, 'LAT', people, left)
            emg_list = np.asarray([emg_BIC, emg_TRI, emg_ANT, emg_POS, emg_PEC, emg_LAT])
            t_list = np.asarray([t1, t2, t3, t4, t5, t6])
        else:
            # emg_rect_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed',
            #                   'DelPos', 'BicLong', 'BicSho', 'TriLong', 'TriLat',
            #                   'BRA', 'BRD', 'LD', 'TerMaj', 'TerMin',
            #                   'Infra', 'Supra', 'Cora']
            emg_list = []
            t_list = []
            for i in range(len(musc_label)):
                [emg_rect, t] = emg_rectification(emg[:, i + 1], fs, musc_label[i], people, left)
                emg_list.append(emg_rect)
                t_list.append(t)
            emg_list = np.asarray(emg_list)
            t_list = np.asarray(t_list)
        return [emg_list, t_list]
    elif sport == 'deadlift':
        emg_rect_label = ['TA', 'GasLat', 'GasMed', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'Sem', 'BF',
                          'GMax', 'GMed', 'PsoMaj', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
        emg_list = []
        t_list = []
        for i in range(len(emg_rect_label)):
            [emg_rect, t] = emg_rectification(emg[:, i + 1], fs, emg_rect_label[i], people, left)
            emg_list.append(emg_rect)
            t_list.append(t)
        emg_list = np.asarray(emg_list)
        t_list = np.asarray(t_list)
        if joint_idx == 'twist':
            emg_list = emg_list[-7:]
            t_list = t_list[-7:]
        elif joint_idx == 'hip':
            emg_list = np.concatenate(([emg_list[4]], emg_list[6:13]))
            t_list = np.concatenate(([t_list[4]], t_list[6:13]))
        elif joint_idx == 'knee':
            emg_list = np.concatenate((emg_list[:6], emg_list[8:10]))
            t_list = np.concatenate((t_list[:6], t_list[8:10]))
        return [emg_list, t_list]
    else:
        print('No such sport.')


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
        file_folder = 'files/bench press/chenzui/'
        files = [file_folder + '4.npy'] * 6
        sport_label = 'bench_press'
        t_delta_joi = [10.266] * 6
        t_delta_emg = [5.916] * 6
        timestep_emg = [
            # [16.966, 20.432, 20.432, 21.999],
            [21.999, 26.499, 26.499, 27.898],
            [28.632, 32.732, 32.732, 34.031],
            [34.498, 38.031, 38.031, 39.365],
            # [40.231, 43.664, 43.664, 46.431],
            [46.797, 50.997, 50.997, 53.364],
            [54.164, 57.564, 57.564, 60.33],
            [60.33, 64.363, 64.363, 67.296],
            # [67.896, 72.129, 72.129, 74.263],
            # [74.429, 78.929, 78.929, 80.862]
        ]  # 4kg
    elif label == 'bp-5.5kg':
        file_folder = 'files/bench press/chenzui/'
        files = [file_folder + '5.5.npy'] * 9
        sport_label = 'bench_press'
        t_delta_emg = [5.655] * 9
        t_delta_joi = [6.467] * 9
        timestep_emg = [
            [16.266, 20.099, 20.099, 22.266],
            [22.699, 26.899, 26.899, 29.432],
            [29.432, 34.882, 34.882, 37.298],
            [37.932, 42.365, 42.365, 44.565],
            [45.131, 49.164, 49.164, 51.664],
            [52.131, 56.097, 56.097, 58.697],
            [58.697, 62.897, 62.897, 65.097],
            [65.497, 69.00, 69.00, 71.296],
            [72.596, 75.00, 75.00, 77.463],
            # [77.463, 80.729, 80.729, 83.562]
        ]
    elif label == 'bp-6.5kg':
        file_folder = 'files/bench press/chenzui/'
        files = [file_folder + '6.5.npy'] * 8
        sport_label = 'bench_press'
        t_delta_emg = [6.8905] * 8
        t_delta_joi = [5.833] * 8
        timestep_emg = [
            [13.533, 16.166, 16.166, 18.066],
            [18.533, 21.766, 21.766, 23.866],
            [24.333, 27.632, 27.632, 29.999],
            [30.199, 34.299, 34.299, 36.132],
            [36.132, 40.165, 40.165, 42.398],
            [42.998, 46.265, 46.265, 49.098],
            [49.565, 52.964, 52.964, 55.264],
            [55.831, 58.764, 58.764, 61.164],
            # [61.597, 64.397, 64.397, 66.73]
        ]  # 4kg
    elif label == 'bp-7kg':
        file_folder = 'files/bench press/chenzui/'
        files = [file_folder + '7.npy'] * 10
        sport_label = 'bench_press'
        t_delta_emg = [5.6485] * 10
        t_delta_joi = [6.1] * 10
        timestep_emg = [
            [15.566, 18.732, 18.732, 20.532],
            [20.532, 24.065, 24.065, 25.899],
            [26.232, 29.966, 29.966, 32.065],
            [32.199, 35.832, 35.832, 38.132],
            [38.132, 42.065, 42.065, 44.598],
            [44.931, 48.265, 48.265, 50.531],
            [50.531, 54.831, 54.831, 56.731],
            [57.197, 61.047, 61.047, 63.063],
            [63.63, 66.63, 66.63, 69.13],
            [69.43, 73.497, 73.497, 75.496]
        ]
    elif label == 'bp-9.5kg':
        file_folder = 'files/bench press/chenzui/'
        files = [file_folder + '9.5.npy'] * 9
        sport_label = 'bench_press'
        t_delta_emg = [6.628] * 9
        t_delta_joi = [5.633] * 9
        timestep_emg = [
            [16.599, 18.965, 18.965, 20.299],
            [20.932, 23.465, 23.465, 24.865],
            [25.232, 27.765, 27.765, 29.131],
            [29.598, 32.065, 32.065, 33.531],
            [33.664, 36.764, 36.764, 38.231],
            [38.431, 41.464, 41.464, 42.897],
            [43.597, 46.047, 46.047, 47.364],
            [48.064, 50.397, 50.397, 51.897],
            [53.064, 55.23, 55.23, 56.83],
            # [57.497, 60.28, 60.28, 61.463]
        ]
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
        muscles = [[([]) for _ in range(len(measured_muscle_idx))] for _ in range(data_set_number)]  # number of muscle
    for k in range(data_set_number):
        for j in range(len(measured_muscle_idx)):
            for i in range(int(len(timestep_emg[k]) / 2)):
                muscles[k][j].append(resample_by_len(emg_all[k][j, find_nearest_idx(t_emg_all[k], timestep_emg[k][
                    2 * i]): find_nearest_idx(t_emg_all[k], timestep_emg[k][2 * i + 1])], unified_len))

    muscles = np.asarray(muscles)

    if need_all_muscle is True:
        data = [([]) for _ in range(len(muscle_idx))]
    else:
        data = [([]) for _ in range(len(measured_muscle_idx))]
    for k in range(len(measured_muscle_idx)):
        for i in range(data_set_number):
            data[k].append(np.concatenate([muscles[i, k, 0, :], muscles[i, k, 1, :]]))

    if need_all_muscle is True:
        if elbow_muscle is True:
            allk = [11, 3, 3, 7, 5, 5]
            alli = [6, 17, 20, 23, 30, 35]
            allc = [0, 1, 2, 3, 4, 5]
        else:
            allk = [3, 3, 7, 5, 5]
            alli = [6, 9, 12, 19, 24]
            allc = [1, 2, 3, 4, 5]
        for i in range(len(allk)):
            for k in range(allk[i]):
                idx = alli[i]
                conum = allc[i]
                data[idx + k] = data[conum]
        if elbow_muscle is True:
            a1, a2 = elbow_emg(data[0], data[1])
            for j in range(3):
                data[14 + j] = a1  # brachiorad
            for j in range(7):
                data[7 + j] = a2  # brachialis
    data = np.asarray(data)

    data_mean = np.ones([len(muscle_idx), data.shape[2]])
    data_std = np.ones([len(muscle_idx), data.shape[2]])
    data_trend_u = np.ones([len(muscle_idx), data.shape[2] - 1])
    data_trend_d = np.ones([len(muscle_idx), data.shape[2] - 1])
    for i in range(len(muscle_idx)):
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
        data_trend_u[i] = np.asarray(
            [(np.max(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray(
            [(np.min(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
    # if need_all_muscle is True:
    #     if elbow_muscle is True:
    #         allk = [11, 3, 3, 7, 5, 5]
    #         alli = [6, 17, 20, 23, 30, 35]
    #         allc = [0, 1, 2, 3, 4, 5]
    #     else:
    #         allk = [3, 3, 7, 5, 5]
    #         alli = [6, 9, 12, 19, 24]
    #         allc = [1, 2, 3, 4, 5]
    #     for i in range(len(allk)):
    #         for k in range(allk[i]):
    #             idx = alli[i]
    #             conum = allc[i]
    #             data_mean[idx + k] = data_mean[conum]
    #             data_std[idx + k] = data_std[conum]
    #             data_trend_u[idx + k] = data_trend_u[conum]
    #             data_trend_d[idx + k] = data_trend_d[conum]
    #     if elbow_muscle is True:
    #         a1, a2 = elbow_emg(emg1, emg2)

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
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
        for j in range(len(measured_muscle_idx)):
            plt.subplot(len(measured_muscle_idx), 1, j + 1)
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
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
        for j in range(len(measured_muscle_idx)):
            plt.subplot(len(measured_muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
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
        for j in range(len(measured_muscle_idx)):
            plt.subplot(len(measured_muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j], color='papayawhip')
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
    num = num + 1
    plt.savefig('emg_{}.png'.format(num))

    if need_all_muscle is True and elbow_muscle is True:
        m = ['Brachialis', 'Brachiorad', 'Biceps', 'Triceps']
        k = [7, 14, 0, 1]
        plt.figure(figsize=(6, 7.7))
        for j in range(len(m)):
            plt.subplot(len(m), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[k[j]], 2 * data_std[k[j]])
            plt.ylabel(m[j], weight='bold')
        plt.figure(figsize=(6, 7.7))
        for j in range(len(m)):
            plt.subplot(len(m), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[k[j]], 2 * data_std[k[j]], color='papayawhip')
            for i in range(data.shape[1]):
                plt.plot(data[k[j], i, :], color=color[i % len(color)])
            plt.ylabel(m[j], weight='bold')

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
    elif label == 'bp-5.5kg':
        np.save('emg/bp-chenzui_mean_5.5kg', data_mean)
        np.save('emg/bp-chenzui_std_5.5kg', data_std)
        np.save('emg/bp-chenzui_trend_u_5.5kg', data_trend_u)
        np.save('emg/bp-chenzui_trend_d_5.5kg', data_trend_d)
    elif label == 'bp-6.5kg':
        np.save('emg/bp-chenzui_mean_6.5kg', data_mean)
        np.save('emg/bp-chenzui_std_6.5kg', data_std)
        np.save('emg/bp-chenzui_trend_u_6.5kg', data_trend_u)
        np.save('emg/bp-chenzui_trend_d_6.5kg', data_trend_d)
    elif label == 'bp-7kg':
        np.save('emg/bp-chenzui_mean_7kg', data_mean)
        np.save('emg/bp-chenzui_std_7kg', data_std)
        np.save('emg/bp-chenzui_trend_u_7kg', data_trend_u)
        np.save('emg/bp-chenzui_trend_d_7kg', data_trend_d)
    elif label == 'bp-9.5kg':
        np.save('emg/bp-chenzui_mean_9.5kg', data_mean)
        np.save('emg/bp-chenzui_std_9.5kg', data_std)
        np.save('emg/bp-chenzui_trend_u_9.5kg', data_trend_u)
        np.save('emg/bp-chenzui_trend_d_9.5kg', data_trend_d)


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

    data_mean = np.ones([data.shape[0], data.shape[2]])
    data_std = np.ones([data.shape[0], data.shape[2]])
    data_trend_u = np.ones([data.shape[0], data.shape[2] - 1])
    data_trend_d = np.ones([data.shape[0], data.shape[2] - 1])
    for i in range(len(muscle_idx)):  # three muscles
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
        data_trend_u[i] = np.asarray(
            [(np.max(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray(
            [(np.min(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    musc_label = ['Biceps', 'Triceps', 'Anterior', 'Medius', 'Pectoralis', 'Latissimus']
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


def calculate_other_emg_distribution(label='yt-bp-20kg'):
    unified_len = emg_lift_len
    fs = 1000
    num = 0
    if date == '240408':
        if label == 'yt-bp-20kg':
            rep = 10
            file_folder = 'files/bench press/muscle-6/yuetian/0408/Test EMG/'
            files = [file_folder + 'bptest 2024_04_08 19_25_06.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [5.1] * rep
            t_delta_joi = [3.3] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-30kg':
            rep = 8
            file_folder = 'files/bench press/yuetian/0408/Test EMG/'
            files = [file_folder + 'bptest 2024_04_08 19_29_21.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [3.56] * rep
            t_delta_joi = [4.816] * rep
            timestep_emg = [
                [9.482, 11.199, 11.199, 12.299],
                [12.699, 14.532, 14.532, 15.565],
                [16.249, 17.949, 17.949, 19.049],
                [19.499, 21.332, 21.332, 22.465],
                [22.898, 24.965, 24.965, 25.965],
                [26.198, 28.415, 28.415, 29.398],
                [29.631, 31.465, 31.465, 32.415],
                [32.648, 34.731, 34.731, 35.748]
            ]
        elif label == 'yt-bp-40kg':
            rep = 8
            file_folder = 'files/bench press/yuetian/0408/Test EMG/'
            files = [file_folder + 'bptest 2024_04_08 19_33_53.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [3.217] * rep
            t_delta_joi = [5.633] * rep
            timestep_emg = [
                [11.616, 13.866, 13.866, 14.849],
                [15.599, 17.283, 17.283, 18.133],
                [18.633, 20.266, 20.266, 21.182],
                [21.632, 23.282, 23.282, 24.216],
                [24.399, 26.116, 26.116, 26.999],
                [27.199, 28.749, 28.749, 29.515],
                [29.882, 31.249, 31.249, 32.049],
                [32.049, 33.499, 33.499, 34.798]
            ]
        elif label == 'yt-bp-50kg':
            rep = 8
            file_folder = 'files/bench press/yuetian/0408/Test EMG/'
            files = [file_folder + 'bptest 2024_04_08 19_35_47.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [2.23] * rep
            t_delta_joi = [4.617] * rep
            timestep_emg = [
                [10.566, 12.333, 12.333, 13.25],
                [13.583, 15.233, 15.233, 16.183],
                [16.433, 17.566, 17.566, 18.599],
                [18.833, 19.899, 19.899, 20.833],
                [21.099, 21.966, 21.966, 23.099],
                [23.099, 24.249, 24.249, 25.416],
                [25.599, 26.899, 26.899, 27.666],
                [27.982, 29.216, 29.216, 30.099]
            ]
        elif label == 'yt-bp-60kg':
            rep = 10
            file_folder = 'files/bench press/yuetian/0408/Test EMG/'
            files = [file_folder + 'bptest 2024_04_08 19_38_03.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [3.273] * rep
            t_delta_joi = [5.317] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-all':
            rep1 = 10
            rep2 = 8
            rep3 = 8
            rep4 = 8
            rep5 = 10
            file_folder = 'files/bench press/yuetian/0408/Test EMG/'
            files = [[file_folder + 'bptest 2024_04_08 19_25_06.xlsx'] * rep1 +
                     [file_folder + 'bptest 2024_04_08 19_29_21.xlsx'] * rep2 +
                     [file_folder + 'bptest 2024_04_08 19_33_53.xlsx'] * rep3 +
                     [file_folder + 'bptest 2024_04_08 19_35_47.xlsx'] * rep4 +
                     [file_folder + 'bptest 2024_04_08 19_38_03.xlsx'] * rep5][0]
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [[5.1] * rep1 + [3.56] * rep2 + [3.217] * rep3 + [2.23] * rep4 + [3.273] * rep5][0]
            t_delta_joi = [[3.3] * rep1 + [4.816] * rep2 + [5.633] * rep3 + [4.617] * rep4 + [5.317] * rep5][0]
            timestep_emg = [
                [8.483, 9.8, 9.8, 10.616],
                [10.616, 11.583, 11.583, 12.316],
                [12.316, 13.3, 13.3, 13.983],
                [13.983, 14.699, 14.699, 15.316],
                [15.316, 15.949, 15.949, 16.466],
                [16.466, 17.149, 17.149, 17.749],
                [17.749, 18.366, 18.366, 18.916],
                [18.916, 19.499, 19.499, 20.083],
                [20.083, 20.666, 20.666, 21.199],
                [21.199, 21.749, 21.749, 22.332],
                [9.482, 11.199, 11.199, 12.299],
                [12.699, 14.532, 14.532, 15.565],
                [16.249, 17.949, 17.949, 19.049],
                [19.499, 21.332, 21.332, 22.465],
                [22.898, 24.965, 24.965, 25.965],
                [26.198, 28.415, 28.415, 29.398],
                [29.631, 31.465, 31.465, 32.415],
                [32.648, 34.731, 34.731, 35.748],
                [11.616, 13.866, 13.866, 14.849],
                [15.599, 17.283, 17.283, 18.133],
                [18.633, 20.266, 20.266, 21.182],
                [21.632, 23.282, 23.282, 24.216],
                [24.399, 26.116, 26.116, 26.999],
                [27.199, 28.749, 28.749, 29.515],
                [29.882, 31.249, 31.249, 32.049],
                [32.049, 33.499, 33.499, 34.798],
                [10.566, 12.333, 12.333, 13.25],
                [13.583, 15.233, 15.233, 16.183],
                [16.433, 17.566, 17.566, 18.599],
                [18.833, 19.899, 19.899, 20.833],
                [21.099, 21.966, 21.966, 23.099],
                [23.099, 24.249, 24.249, 25.416],
                [25.599, 26.899, 26.899, 27.666],
                [27.982, 29.216, 29.216, 30.099],
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
        else:
            print('No label', label, 'in date', date)
            return 0
    elif date == '240604':
        if label == 'yt-bp-20kg':
            rep = 10
            file_folder = 'files/bench press/muscle-18/yuetian/emg/'
            files = [file_folder + 'test 2024_06_04 16_16_22.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [0] * rep
            t_delta_joi = [0] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-30kg':
            rep = 10
            file_folder = 'files/bench press/muscle-18/yuetian/emg/'
            files = [file_folder + 'test 2024_06_04 16_21_47.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [0] * rep
            t_delta_joi = [0] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-40kg':
            rep = 10
            file_folder = 'files/bench press/muscle-18/yuetian/emg/'
            files = [file_folder + 'test 2024_06_04 16_23_42.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [0] * rep
            t_delta_joi = [0] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-50kg':
            rep = 10
            file_folder = 'files/bench press/muscle-18/yuetian/emg/'
            files = [file_folder + 'test 2024_06_04 16_26_07.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [0] * rep
            t_delta_joi = [0] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-60kg':
            rep = 10
            file_folder = 'files/bench press/muscle-18/yuetian/emg/'
            files = [file_folder + 'test 2024_06_04 16_29_21.xlsx'] * rep
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [0] * rep
            t_delta_joi = [0] * rep
            timestep_emg = [
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
        elif label == 'yt-bp-all':
            rep = [10] * 5
            file_folder = 'files/bench press/muscle-18/yuetian/emg/'
            files = ([file_folder + 'test 2024_06_04 16_16_22.xlsx'] * rep[0] +
                     [file_folder + 'test 2024_06_04 16_21_47.xlsx'] * rep[1] +
                     [file_folder + 'test 2024_06_04 16_23_42.xlsx'] * rep[2] +
                     [file_folder + 'test 2024_06_04 16_26_07.xlsx'] * rep[3] +
                     [file_folder + 'test 2024_06_04 16_29_21.xlsx'] * rep[4])
            sport_label = 'bench_press'
            people = 'yuetian'
            t_delta_emg = [0] * sum(rep)
            t_delta_joi = [0] * sum(rep)
            timestep_emg = [
                [6.650, 8.133, 8.133, 8.983],
                [9.583, 10.766, 10.766, 11.583],
                [12.032, 13.149, 13.149, 13.949],
                [14.316, 15.599, 15.599, 16.549],
                [16.732, 17.956, 17.956, 19.032],
                [19.682, 20.715, 20.715, 21.449],
                [22.015, 23.082, 23.082, 23.865],
                [23.998, 25.315, 25.315, 26.182],
                [26.565, 27.698, 27.698, 28.615],
                [28.982, 30.132, 30.132, 31.064],
                [11.449, 12.633, 12.633, 13.383],
                [13.449, 14.399, 14.399, 15.149],
                [15.366, 16.182, 16.182, 16.882],
                [17.032, 17.916, 17.916, 18.749],
                [18.749, 19.566, 19.566, 20.249],
                [20.315, 21.032, 21.032, 21.682],
                [21.682, 22.415, 22.415, 23.115],
                [23.115, 23.765, 23.765, 24.482],
                [24.482, 25.215, 25.215, 25.915],
                [25.915, 26.765, 26.765, 27.365],
                [7.600, 8.783, 8.783, 9.517],
                [9.683, 10.483, 10.483, 11.117],
                [11.200, 11.867, 11.867, 12.500],
                [12.567, 13.300, 13.300, 13.900],
                [13.900, 14.550, 14.550, 15.183],
                [15.216, 16.200, 16.200, 16.783],
                [16.783, 17.383, 17.383, 17.949],
                [17.949, 18.533, 18.533, 19.100],
                [19.100, 19.683, 19.683, 20.216],
                [20.216, 20.833, 20.833, 21.433],
                [7.850, 8.783, 8.783, 9.350],
                [9.666, 10.250, 10.250, 10.800],
                [11.066, 11.650, 11.650, 12.216],
                [12.400, 13.000, 13.000, 13.566],
                [13.733, 14.266, 14.266, 14.849],
                [14.916, 15.533, 15.533, 16.216],
                [16.216, 16.816, 16.816, 17.399],
                [17.499, 18.033, 18.033, 18.599],
                [18.666, 19.183, 19.183, 19.783],
                [19.783, 20.333, 20.333, 21.116],
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
        else:
            print('No label', label, 'in date', date)
            return 0
    elif label == 'yt-dl-35kg':
        rep = 6
        file_folder = 'files/deadlift/yuetian/emg/'
        files = [file_folder + 'test 2024_05_17 16_27_52.xlsx'] * rep
        sport_label = 'deadlift'
        people = 'yuetian'
        t_delta_emg = [0] * rep
        t_delta_joi = [0] * rep
        timestep_emg = [
            [7.649, 8.949, 8.949, 10.532],
            [11.149, 12.333, 12.333, 14.099],
            [14.632, 15.665, 15.666, 17.215],
            [17.799, 18.898, 18.898, 20.566],
            [20.933, 21.933, 21.933, 23.366],
            [23.933, 25.049, 25.049, 26.132]
        ]
    elif label == 'yt-dl-45kg':
        rep = 6
        file_folder = 'files/deadlift/yuetian/emg/'
        files = [file_folder + 'test 2024_05_17 16_30_59.xlsx'] * rep
        sport_label = 'deadlift'
        people = 'yuetian'
        t_delta_emg = [0] * rep
        t_delta_joi = [0] * rep
        timestep_emg = [
            [8.783, 9.5, 9.5, 10.65],
            [11.4, 11.983, 11.983, 13.15],
            [13.933, 14.833, 14.833, 15.999],
            [16.75, 17.75, 17.75, 18.633],
            [19.416, 20.366, 20.366, 21.149],
            [21.883, 22.533, 22.533, 23.566]
        ]
    elif label == 'yt-dl-65kg':
        rep = 5
        file_folder = 'files/deadlift/yuetian/emg/'
        files = [file_folder + 'test 2024_05_17 16_37_38.xlsx'] * rep
        sport_label = 'deadlift'
        people = 'yuetian'
        t_delta_emg = [0] * rep
        t_delta_joi = [0] * rep
        timestep_emg = [
            [7.266, 8.166, 8.166, 9.316],
            [10.616, 11.466, 11.466, 12.732],
            [13.782, 14.466, 14.466, 15.682],
            [16.615, 17.399, 17.399, 18.349],
            [19.282, 20.039, 20.039, 21.049]
        ]
    elif label == 'yt-dl-75kg':
        rep = 6
        file_folder = 'files/deadlift/yuetian/emg/'
        files = [file_folder + 'test 2024_05_17 16_41_21.xlsx'] * rep
        # sport_label = 'deadlift'
        people = 'yuetian'
        t_delta_emg = [0] * rep
        t_delta_joi = [0] * rep
        timestep_emg = [
            [6.566, 7.283, 7.283, 8.333],
            [9.133, 9.789, 9.789, 10.766],
            [11.483, 12.099, 12.099, 13.149],
            [13.916, 14.536, 14.536, 15.565],
            [16.199, 16.966, 16.966, 17.816],
            [18.482, 19.032, 19.032, 20.065]
        ]
    elif label == 'yt-dl-all':
        rep = [6, 6, 5, 6]
        file_folder = 'files/deadlift/yuetian/emg/'
        files = ([file_folder + 'test 2024_05_17 16_27_52.xlsx'] * rep[0] +
                 [file_folder + 'test 2024_05_17 16_30_59.xlsx'] * rep[1] +
                 [file_folder + 'test 2024_05_17 16_37_38.xlsx'] * rep[2] +
                 [file_folder + 'test 2024_05_17 16_41_21.xlsx'] * rep[3])
        sport_label = 'deadlift'
        people = 'yuetian'
        t_delta_emg = [0] * sum(rep)
        t_delta_joi = [0] * sum(rep)
        timestep_emg = [
            [7.649, 8.949, 8.949, 10.532],
            [11.149, 12.333, 12.333, 14.099],
            [14.632, 15.665, 15.666, 17.215],
            [17.799, 18.898, 18.898, 20.566],
            [20.933, 21.933, 21.933, 23.366],
            [23.933, 25.049, 25.049, 26.132],
            [8.783, 9.5, 9.5, 10.65],
            [11.4, 11.983, 11.983, 13.15],
            [13.933, 14.833, 14.833, 15.999],
            [16.75, 17.75, 17.75, 18.633],
            [19.416, 20.366, 20.366, 21.149],
            [21.883, 22.533, 22.533, 23.566],
            [7.266, 8.166, 8.166, 9.316],
            [10.616, 11.466, 11.466, 12.732],
            [13.782, 14.466, 14.466, 15.682],
            [16.615, 17.399, 17.399, 18.349],
            [19.282, 20.039, 20.039, 21.049],
            [6.566, 7.283, 7.283, 8.333],
            [9.133, 9.789, 9.789, 10.766],
            [11.483, 12.099, 12.099, 13.149],
            [13.916, 14.536, 14.536, 15.565],
            [16.199, 16.966, 16.966, 17.816],
            [18.482, 19.032, 19.032, 20.065]
        ]
    elif label == 'kh-bp-20kg':
        rep = 10
        file_folder = 'files/bench press/muscle-18/kehan/emg/'
        files = [file_folder + 'test 2024_07_31 11_25_10.csv'] * rep
        sport_label = 'bench_press'
        people = 'kehan'
        t_delta_emg = [0] * rep
        t_delta_joi = [0] * rep
        timestep_emg = [
            [6.667, 7.783, 7.783, 8.233],
            [8.966, 9.766, 9.766, 10.350],
            [11.200, 11.816, 11.816, 12.483],
            [13.300, 14.233, 14.233, 14.649],
            [15.366, 16.316, 16.316, 16.733],
            [17.483, 18.416, 18.416, 18.966],
            [19.566, 20.499, 20.499, 21.049],
            [21.649, 22.749, 22.749, 23.116],
            [23.849, 24.916, 24.916, 25.349],
            [26.266, 27.249, 27.249, 27.799]
        ]
    elif label == 'kh-bp-30kg':
        rep = 10
        file_folder = 'files/bench press/muscle-18/kehan/emg/'
        files = [file_folder + 'test 2024_07_31 11_27_28.csv'] * rep
        sport_label = 'bench_press'
        people = 'kehan'
        t_delta_emg = [0] * rep
        t_delta_joi = [0] * rep
        timestep_emg = [
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
    elif label == 'kh-bp-all':
        rep = [10, 10, 8, 5, 8]
        file_folder = 'files/bench press/muscle-18/kehan/emg/'
        files = ([file_folder + 'test 2024_07_31 11_25_10.csv'] * rep[0] +
                 [file_folder + 'test 2024_07_31 11_27_28.csv'] * rep[1] +
                 [file_folder + 'test 2024_07_31 11_29_38.csv'] * rep[2] +
                 [file_folder + 'test 2024_07_31 11_36_27.csv'] * rep[3] +
                 [file_folder + 'test 2024_07_31 11_42_15.csv'] * rep[4])
        sport_label = 'bench_press'
        people = 'kehan'
        t_delta_emg = [0] * sum(rep)
        t_delta_joi = [0] * sum(rep)
        timestep_emg = [
            [6.667, 7.783, 7.783, 8.233],
            [8.966, 9.766, 9.766, 10.350],
            [11.200, 11.816, 11.816, 12.483],
            [13.300, 14.233, 14.233, 14.649],
            [15.366, 16.316, 16.316, 16.733],
            [17.483, 18.416, 18.416, 18.966],
            [19.566, 20.499, 20.499, 21.049],
            [21.649, 22.749, 22.749, 23.116],
            [23.849, 24.916, 24.916, 25.349],
            [26.266, 27.249, 27.249, 27.799],
            [10.466, 11.616, 11.616, 12.283],
            [13.083, 14.100, 14.100, 14.666],
            [15.399, 16.516, 16.516, 17.066],
            [17.766, 18.749, 18.749, 19.266],
            [19.966, 21.099, 21.099, 21.649],
            [22.616, 23.682, 23.682, 24.216],
            [25.149, 26.199, 26.199, 26.732],
            [27.466, 28.482, 28.482, 29.032],
            [29.799, 30.849, 30.849, 31.399],
            [32.165, 33.182, 33.182, 33.732],
            [16.866, 17.966, 17.966, 18.566],
            [19.132, 19.949, 19.949, 20.516],
            [21.066, 22.182, 22.182, 22.815],
            [23.215, 24.315, 24.315, 24.932],
            [25.465, 26.615, 26.615, 27.265],
            [27.765, 28.849, 28.849, 29.498],
            [30.032, 31.082, 31.082, 31.765],
            [32.448, 33.448, 33.448, 34.148],
            [7.717, 9.017, 9.017, 9.767],
            [10.400, 11.483, 11.483, 12.266],
            [12.916, 14.100, 14.100, 14.950],
            [15.583, 16.750, 16.750, 17.533],
            [18.149, 19.466, 19.466, 20.449],
            [9.083, 10.066, 10.066, 10.716],
            [11.366, 12.183, 12.183, 12.749],
            [13.599, 14.416, 14.416, 15.066],
            [16.099, 17.049, 17.049, 17.733],
            [18.616, 19.482, 19.482, 20.166],
            [20.849, 21.732, 21.732, 22.499],
            [23.316, 24.299, 24.299, 25.165],
            [25.999, 27.082, 27.082, 27.982]
        ]
    else:
        print('No label:', label)
        return 0

    data_set_number = len(files)
    emg_all = [([]) for _ in range(data_set_number)]
    t_emg_all = [([]) for _ in range(data_set_number)]
    for i in range(data_set_number):
        if '.csv' in files[i]:
            emg_all[i] = from_csv_to_emg(files[i])
        else:
            emg_all[i] = np.asarray(pd.read_excel(files[i]))
        [emg_all[i], t_emg_all[i]] = emg_file_progressing(emg_all[i], fs, people, sport_label)
        t_emg_all[i] = t_emg_all[i] - t_delta_emg[i] + t_delta_joi[i]

    if sport_label == 'bench_press' or sport_label == 'deadlift':
        muscles = [[([]) for _ in range(len(measured_muscle_idx))] for _ in range(data_set_number)]  # number of muscle
    for k in range(data_set_number):
        for j in range(len(measured_muscle_idx)):
            for i in range(int(len(timestep_emg[k]) / 2)):
                muscles[k][j].append(resample_by_len(emg_all[k][j, find_nearest_idx(t_emg_all[k], timestep_emg[k][
                    2 * i]): find_nearest_idx(t_emg_all[k], timestep_emg[k][2 * i + 1])], unified_len))

    muscles = np.asarray(muscles)

    if need_all_muscle is True:
        data = [([]) for _ in range(len(muscle_idx))]
    else:
        data = [([]) for _ in range(len(measured_muscle_idx))]
    for k in range(len(measured_muscle_idx)):
        for i in range(data_set_number):
            data[k].append(np.concatenate([muscles[i, k, 0, :], muscles[i, k, 1, :]]))

    if sport_label == 'bench_press' or sport_label == 'deadlift':
        allk = related_muscle_num
        alli = related_muscle_idx
        allc = list(range(len(related_muscle_num)))
        for i in range(len(allk)):
            for k in range(allk[i]):
                idx = alli[i]
                conum = allc[i]
                data[idx + k] = data[conum]
    data = np.asarray(data)

    data_mean = np.ones([len(muscle_idx), data.shape[2]])
    data_std = np.ones([len(muscle_idx), data.shape[2]])
    data_trend_u = np.ones([len(muscle_idx), data.shape[2] - 1])
    data_trend_d = np.ones([len(muscle_idx), data.shape[2] - 1])
    for i in range(len(muscle_idx)):
        data_mean[i] = np.asarray([np.mean(data[i, :, j]) for j in range(data.shape[2])])
        data_std[i] = np.asarray([np.std(data[i, :, j]) for j in range(data.shape[2])])
        data_trend_u[i] = np.asarray(
            [(np.max(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])
        data_trend_d[i] = np.asarray(
            [(np.min(data[i, :, j + 1] - data[i, :, j]) / 0.01) for j in range(data.shape[2] - 1)])

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    if len(measured_muscle_idx) <= 10:
        plt.figure(figsize=(6, 7.7))
        for j in range(len(measured_muscle_idx)):
            plt.subplot(len(measured_muscle_idx), 1, j + 1)
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        num = num + 1
        plt.savefig('emg_{}.png'.format(num))
    else:
        plt.figure(figsize=(10, 7.7))
        for j in range(len(measured_muscle_idx)):
            plt.subplot(round(len(measured_muscle_idx) / 2), 2, j + 1)
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        num = num + 1
        plt.savefig('emg_{}.png'.format(num))

    if len(measured_muscle_idx) <= 10:
        plt.figure(figsize=(6, 7.7))
        for j in range(len(measured_muscle_idx)):
            plt.subplot(len(measured_muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
        num = num + 1
        plt.savefig('emg_{}.png'.format(num))
    else:
        plt.figure(figsize=(10, 7.7))
        for j in range(len(measured_muscle_idx)):
            plt.subplot(round(len(measured_muscle_idx) / 2), 2, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
        num = num + 1
        plt.savefig('emg_{}.png'.format(num))

    if len(measured_muscle_idx) <= 10:
        plt.figure(figsize=(6, 7.7))
        for j in range(len(measured_muscle_idx)):
            plt.subplot(len(measured_muscle_idx), 1, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j], color='papayawhip')
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
        num = num + 1
        plt.savefig('emg_{}.png'.format(num))
    else:
        plt.figure(figsize=(10, 7.7))
        for j in range(len(measured_muscle_idx)):
            plt.subplot(round(len(measured_muscle_idx) / 2), 2, j + 1)
            plt.errorbar(range(data_mean.shape[1]), data_mean[j], 2 * data_std[j], color='papayawhip')
            for i in range(data.shape[1]):
                plt.plot(data[j, i, :], color=color[i % len(color)])
            plt.ylabel(musc_label[j], weight='bold')
            if j != len(measured_muscle_idx) - 1:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
        plt.xlabel('timestep', weight='bold')
        num = num + 1
        plt.savefig('emg_{}.png'.format(num))

    if label == 'yt-bp-20kg':
        np.save(file_folder + 'mean_20kg', data_mean)
        np.save(file_folder + 'std_20kg', data_std)
        np.save(file_folder + 'trend_u_20kg', data_trend_u)
        np.save(file_folder + 'trend_d_20kg', data_trend_d)
    elif label == 'yt-bp-30kg':
        np.save(file_folder + 'mean_30kg', data_mean)
        np.save(file_folder + 'std_30kg', data_std)
        np.save(file_folder + 'trend_u_30kg', data_trend_u)
        np.save(file_folder + 'trend_d_30kg', data_trend_d)
    elif label == 'yt-bp-40kg':
        np.save(file_folder + 'mean_40kg', data_mean)
        np.save(file_folder + 'std_40kg', data_std)
        np.save(file_folder + 'trend_u_40kg', data_trend_u)
        np.save(file_folder + 'trend_d_40kg', data_trend_d)
    elif label == 'yt-bp-50kg':
        np.save(file_folder + 'mean_50kg', data_mean)
        np.save(file_folder + 'std_50kg', data_std)
        np.save(file_folder + 'trend_u_50kg', data_trend_u)
        np.save(file_folder + 'trend_d_50kg', data_trend_d)
    elif label == 'yt-bp-60kg':
        np.save(file_folder + 'mean_60kg', data_mean)
        np.save(file_folder + 'std_60kg', data_std)
        np.save(file_folder + 'trend_u_60kg', data_trend_u)
        np.save(file_folder + 'trend_d_60kg', data_trend_d)
    elif label == 'yt-bp-all' or label == 'kh-bp-all':
        np.save(file_folder + 'mean_all', data_mean)
        np.save(file_folder + 'std_all', data_std)
        np.save(file_folder + 'trend_u_all', data_trend_u)
        np.save(file_folder + 'trend_d_all', data_trend_d)
    elif label == 'yt-dl-35kg':
        np.save(file_folder + 'mean_dl_35kg', data_mean)
        np.save(file_folder + 'std_dl_35kg', data_std)
        np.save(file_folder + 'trend_u_dl_35kg', data_trend_u)
        np.save(file_folder + 'trend_d_dl_35kg', data_trend_d)
    elif label == 'yt-dl-65kg':
        np.save(file_folder + 'mean_dl_65kg', data_mean)
        np.save(file_folder + 'std_dl_65kg', data_std)
        np.save(file_folder + 'trend_u_dl_65kg', data_trend_u)
        np.save(file_folder + 'trend_d_dl_65kg', data_trend_d)
    elif label == 'yt-dl-75kg':
        np.save(file_folder + 'mean_dl_75kg', data_mean)
        np.save(file_folder + 'std_dl_75kg', data_std)
        np.save(file_folder + 'trend_u_dl_75kg', data_trend_u)
        np.save(file_folder + 'trend_d_dl_75kg', data_trend_d)
    elif label == 'yt-dl-all':
        np.save(file_folder + 'mean_dl_all', data_mean)
        np.save(file_folder + 'std_dl_all', data_std)
        np.save(file_folder + 'trend_u_dl_all', data_trend_u)
        np.save(file_folder + 'trend_d_dl_all', data_trend_d)


if __name__ == '__main__':
    # calculate_chenzui_emg_distribution(label='bp-9.5kg')
    # calculate_chenzui_emg_distribution(label='bp-7kg')
    # calculate_chenzui_emg_distribution(label='bp-6.5kg')
    # calculate_chenzui_emg_distribution(label='bp-5.5kg')
    # calculate_chenzui_emg_distribution(label='bp-4kg')
    # calculate_lizhuo_emg_distribution(label='1kg')
    calculate_other_emg_distribution(label='kh-bp-30kg')
    # calculate_other_emg_distribution(label='kh-bp-all')
    # calculate_other_emg_distribution(label='yt-bp-20kg')
    # calculate_other_emg_distribution(label='yt-bp-30kg')
    # calculate_other_emg_distribution(label='yt-bp-40kg')
    # calculate_other_emg_distribution(label='yt-bp-50kg')
    # calculate_other_emg_distribution(label='yt-bp-60kg')
    plt.show()
