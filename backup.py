import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy
from scipy import signal
from scipy.optimize import minimize
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd

# from emg_distribution import *
# from test import *
# from require import *


def emg_rectification(x, Fs=1000, code=None, people=None, left=False):
    # Fs 采样频率，在EMG信号中是1000Hz
    NUMPASSES = 3
    LOWPASSRATE1 = 20  # 低通滤波4—10Hz得到包络线
    LOWPASSRATE2 = 450
    Wn1 = LOWPASSRATE1 / Fs
    Wn2 = LOWPASSRATE2 / Fs
    [b, a] = scipy.signal.butter(NUMPASSES, [Wn1, Wn2], 'band')
    x = scipy.signal.filtfilt(b, a, x, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

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

    if people == 'yuetian':  # left
        if code == 'BIC':
            ref = 0.725
        elif code == 'TRI':
            ref = 0.08
        elif code == 'ANT':
            ref = 0.224
        # elif code == 'POS':
        #     ref = 0.23831
        # elif code == 'PEC':
        #     ref = 0.43018
        # elif code == 'LAT':
        #     ref = 0.15176
        # elif code == 'BRA':
        #     ref = 0.67077
        # elif code == 'BRD':
        #     ref = 0.67077
        # else:
        #     ref = 2
        # # ref = 2
        else:
            ref = max(EMGLE)
            print(ref)
    else:
        print('No information of this people.')
    # ref = 1

    normalized_EMG = EMGLE / ref
    y = normalized_EMG
    return [y, t]


def read_emg_mat(file):
    fs = 1000
    mat = scipy.io.loadmat(file)
    mat = mat['TotalDataStruct']
    s1 = mat['S1_Data'][0][0][:, 0]
    s2 = mat['S2_Data'][0][0][:, 0]
    s3 = mat['S3_Data'][0][0][:, 0]
    s4 = mat['S4_Data'][0][0][:, 0]

    [y1, t1] = emg_rectification(s1, fs, 'BIC', 'yuetian')
    [y2, t2] = emg_rectification(s2, fs, 'TRI', 'yuetian')
    [y3, t3] = emg_rectification(s3, fs, 'ANT', 'yuetian')

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(t1, y1)
    # plt.subplot(312)
    # plt.plot(t2, y2)
    # plt.subplot(313)
    # plt.plot(t3, y3)
    # plt.show()

    return y1, y2, y3, t1, t2, t3


def read_weight(file):
    height = pd.read_excel(file)
    t = height['time']
    h1 = -height[' Axis3_pos'] + 0.37
    h2 = -height[' Axis4_pos'] + 0.37
    return h1, h2, t


if __name__ == '__main__':
    folder = 'files/bench press/yuetian/0408'
    h1, h2, t = read_weight(folder + '/20240408robot/30-50Kgchenyuetian/data_1712586336718.xlsx')
    y1, y2, y3, t1, t2, t3 = read_emg_mat(folder + '/new emg/BenchPress30Kg.mat')
    # h1, h2, t = read_weight(folder + '/20240408robot/30-50Kgchenyuetian/data_1712586453766.xlsx')
    # y1, y2, y3, t1, t2, t3 = read_emg_mat(folder + '/new emg/BenchPress40Kg.mat')

    plt.figure()
    plt.subplot(511)
    plt.plot(t, h1)
    plt.subplot(512)
    plt.plot(t, h2)
    plt.subplot(513)
    plt.plot(t1, y1)
    plt.subplot(514)
    plt.plot(t2, y2)
    plt.subplot(515)
    plt.plot(t3, y3)

    # plt.figure()
    # plt.plot(np.linspace(0, 2.5, 2500), y2[17000:19500])
    # plt.plot(np.linspace(0, 2.0, 2000), y2[21500:23500])
    # plt.title('Triceps Brachii', weight='bold')
    # plt.ylabel('Activation', weight='bold')
    # plt.xlabel('Time(s)', weight='bold')
    #
    # plt.figure()
    # plt.plot(np.linspace(0, 1.4, 1400), y2[21800:23200])
    # plt.plot(np.linspace(0, 1.0, 1000), y2[19900:20900])
    # plt.title('Triceps Brachii', weight='bold')
    # plt.ylabel('Activation', weight='bold')
    # plt.xlabel('Time(s)', weight='bold')
    #
    # # plt.figure()
    # # plt.plot(t3[20700:22700], y3[20700:22700])
    #
    # plt.figure()
    # act = np.load('predication.npy')
    # act1 = np.load('predication1.npy')
    # time = np.load('time.npy')
    # plt.plot(time - time[0], np.asarray(act[1, :]))
    # plt.plot(time - time[0], np.asarray(act[2, :]))
    # plt.title('Triceps Brachii', weight='bold')
    # plt.ylabel('Activation', weight='bold')
    # plt.xlabel('Time(s)', weight='bold')
    #
    # plt.figure()
    # act = np.load('predication.npy')
    # # act1 = np.load('predication1.npy')
    # time = np.load('time.npy')
    # plt.plot(time - time[0], np.asarray(act[1, :]))
    # plt.plot(time - time[0], np.asarray(act1[1, :]))
    # plt.title('Triceps Brachii', weight='bold')
    # plt.ylabel('Activation', weight='bold')
    # plt.xlabel('Time(s)', weight='bold')

    plt.figure()
    act = np.load('predication.npy')
    # act1 = np.load('predication1.npy')
    time = np.load('time.npy')
    plt.title('Triceps Brachii', weight='bold')
    plt.subplot(211)
    plt.plot(time - time[0], np.asarray(act[1, :]), color='#1f77b4', label='predication')
    plt.ylabel('Activation', weight='bold')
    plt.legend()
    plt.subplot(212)
    plt.plot(np.linspace(0, 1.2, 1200), y2[21800:23000], color='#ff7f0e', label='measure')
    # plt.plot(time - time[0], np.asarray(act1[1, :]))
    plt.ylabel('Activation', weight='bold')
    plt.xlabel('Time(s)', weight='bold')
    plt.legend()
    plt.show()

