import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
from basic import *


class EMGOneGroup:
    def __init__(self, parameter, file, timestep, label, t_delta_emg=0, t_delta_joi=0):
        self.parameter = parameter
        self.file = file
        self.timestep = timestep
        self.label = label
        self.t_delta_emg = t_delta_emg
        self.t_delta_joi = t_delta_joi

        self._unified_len = parameter.data_length
        self.rep_num = len(self.timestep)
        self.musc_num = len(self.parameter.musc_label)
        self.idx_list = np.arange(self.rep_num) + 1
        self.raw_data = None
        if '.csv' in self.file:
            self.load_emg_from_csv()

        self.time_long = None
        self.activation_long = None
        self.rectification()

        self.data_mean = None
        self.data_std = None
        self.data_trend_u = None
        self.data_trend_d = None
        self.activation = None
        self.distribution()
        # self.plot_distribution(figure_num=[2, 3])

    def __repr__(self):
        return f'EMGData(file={self.file}), data={self.raw_data}'

    def load_emg_from_csv(self):
        states = pd.read_csv(self.file, low_memory=False, skiprows=2)
        raw_data = np.squeeze(np.asarray([states]))
        raw_data = raw_data[3:, 0:self.musc_num + 1].astype(float)  # 前musc_num个肌肉, 包括第一列时间
        # raw_data = raw_data[3:, 1:self.musc_num + 1].astype(float)  # 前musc_num个肌肉, 不包括第一列时间
        # raw_data = np.concatenate([raw_data[3:, 0:13], raw_data[3:, 19:25]], axis=1).astype(float)  # 前12和后6, 包括第一列时间
        # raw_data = np.concatenate([raw_data[3:, 1:13], raw_data[3:, 19:25]], axis=1).astype(float)  # 前12和后6, 不包括第一列时间
        self.raw_data = raw_data

    def rectification(self):
        act_list = []
        t_list = []
        for i in range(self.musc_num):
            [act, t] = self.rectification_one_channel(self.raw_data[:, i + 1], self.parameter.musc_mvc[i])
            act_list.append(act)
            t_list.append(t)
        self.activation_long = np.asarray(act_list)
        t_list = np.asarray(t_list)
        self.time_long = t_list - self.t_delta_emg + self.t_delta_joi

    def rectification_one_channel(self, x, ref):
        # 带通滤波
        nyquist_frequency = self.parameter.emg_fs / 2
        low_cutoff = 20 / nyquist_frequency
        high_cutoff = 450 / nyquist_frequency
        [b, a] = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
        x = scipy.signal.filtfilt(b, a, x)

        # 全波整流
        x_mean = np.mean(x)
        raw = x - x_mean * np.ones_like(x)
        t = np.arange(0, raw.size / self.parameter.emg_fs, 1 / self.parameter.emg_fs)
        EMGFWR = abs(raw)

        # 线性包络 Linear Envelope
        NUMPASSES = 3
        LOWPASSRATE = 4  # 低通滤波2—10Hz得到包络线
        Wn = LOWPASSRATE / (self.parameter.emg_fs / 2)
        [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
        EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

        # normalization
        normalized_EMG = EMGLE / ref
        y = normalized_EMG
        # return [y, t]

        # neural_activation
        neural_activation = np.zeros_like(normalized_EMG)
        for i in range(len(normalized_EMG)):
            if i > 1:
                neural_activation[i] = 2.25 * y[i] - 1 * neural_activation[i - 1] - 0.25 * neural_activation[i - 2]
        u = neural_activation
        return [u, t]

    def distribution(self):
        # 将emg数据按照指定周期截开，并重写为指定长度
        muscles = [[([]) for _ in range(self.musc_num)] for _ in range(self.rep_num)]  # number of muscle
        for k in range(self.rep_num):
            for j in range(self.musc_num):
                for i in range(int(len(self.timestep[k]) / 2)):
                    muscles[k][j].append(resample_by_len(
                        self.activation_long[j, find_nearest_idx(self.time_long, self.timestep[k][2 * i]):
                                                find_nearest_idx(self.time_long, self.timestep[k][2 * i + 1])],
                        self._unified_len))
        muscles = np.asarray(muscles)

        # 将前后半周期合并为一个完整周期
        data = [([]) for _ in range(self.musc_num)]
        for k in range(self.musc_num):
            for i in range(self.rep_num):
                data[k].append(np.concatenate([muscles[i, k, 0, :], muscles[i, k, 1, :]]))

        # 将测量肌肉与模型肌肉对应
        allk = self.parameter.related_muscle_num
        alli = self.parameter.related_muscle_idx
        allc = list(range(len(self.parameter.related_muscle_num)))
        for i in range(len(allk)):
            for k in range(allk[i]):
                idx = alli[i]
                conum = allc[i]
                data[idx + k] = data[conum]
        self.activation = np.asarray(data)

        # distribution
        data_mean = np.ones([self.musc_num, self.activation.shape[2]])
        data_std = np.ones([self.musc_num, self.activation.shape[2]])
        data_trend_u = np.ones([self.musc_num, self.activation.shape[2] - 1])
        data_trend_d = np.ones([self.musc_num, self.activation.shape[2] - 1])
        for i in range(self.musc_num):
            data_mean[i] = np.asarray([np.mean(self.activation[i, :, j]) for j in range(self.activation.shape[2])])
            data_std[i] = np.asarray([np.std(self.activation[i, :, j]) for j in range(self.activation.shape[2])])
            data_trend_u[i] = np.asarray([(np.max(self.activation[i, :, j + 1] - self.activation[i, :, j]) / 0.01)
                                          for j in range(self.activation.shape[2] - 1)])
            data_trend_d[i] = np.asarray([(np.min(self.activation[i, :, j + 1] - self.activation[i, :, j]) / 0.01)
                                          for j in range(self.activation.shape[2] - 1)])
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_trend_u = data_trend_u
        self.data_trend_d = data_trend_d

    def plot_distribution(self, figure_num=None):
        num = 0
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if self.musc_num <= 10:
            column_num = 1
        else:
            column_num = 2

        if column_num == 1:
            figure_size = (6, 7.7)
            adjust_params = {'left': 0.125, 'right': 0.9, 'bottom': 0.11, 'top': 0.88,
                             'wspace': 0.2, 'hspace': 0.2}
        else:
            figure_size = (7, 5.7)
            adjust_params = {'left': 0.109, 'right': 0.962, 'bottom': 0.095, 'top': 0.945,
                             'wspace': 0.26, 'hspace': 0.2}

        if figure_num is None:
            figure_num = [1, 2, 3]
        for fnum in figure_num:
            plt.figure(figsize=figure_size)
            plt.subplots_adjust(**adjust_params)
            for j in range(self.musc_num):
                plt.subplot(math.ceil(self.musc_num / column_num), column_num, j + 1)
                if fnum == 1:
                    for i in range(self.activation.shape[1]):
                        plt.plot(self.activation[j, i, :], color=color[i % len(color)])
                elif fnum == 2:
                    plt.errorbar(range(self.data_mean.shape[1]), self.data_mean[j], 2 * self.data_std[j])
                elif fnum == 3:
                    plt.errorbar(range(self.data_mean.shape[1]), self.data_mean[j], 2 * self.data_std[j],
                                 color='papayawhip')
                    for i in range(self.activation.shape[1]):
                        plt.plot(self.activation[j, i, :], color=color[i % len(color)])

                plt.ylabel(self.parameter.musc_label[j], weight='bold')
                if j < self.musc_num - column_num:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                else:
                    plt.xlabel('timestep', weight='bold')
            num = num + 1
            plt.savefig(self.parameter.result_folder + 'emg_' + self.label + 'kg_{}.png'.format(num))


class EMGData:
    def __init__(self, folder, label, parameter, json_data):
        self.folder = folder
        self.label = label
        self.parameter = parameter
        self.json_data = json_data

        self.data_mean = None
        self.data_std = None
        self.data_trend_u = None
        self.data_trend_d = None

        self.emg_groups = []
        self.rep_num = []
        self.activation = None
        for i in range(len(self.json_data)):
            emg_record = EMGOneGroup(
                parameter=self.parameter,
                file=self.folder + self.json_data[i]['file'],
                timestep=self.json_data[i]['timestep'],
                label=self.label[i]
            )
        # for entry in self.json_data:
        #     emg_record = EMGOneGroup(
        #         parameter=self.parameter,
        #         file=self.folder + entry['file'],
        #         timestep=entry['timestep'],
        #     )
            self.emg_groups.append(emg_record)
            self.rep_num.append(emg_record.rep_num)
            if self.activation is None:
                self.activation = emg_record.activation
            else:
                self.activation = np.concatenate([self.activation, emg_record.activation], axis=1)

        self.group_num = len(self.emg_groups)
        self.musc_num = len(self.parameter.musc_label)
        assert self.group_num == len(self.label)
        self.distribution()

    def distribution(self):
        data_mean = np.ones([self.musc_num, self.activation.shape[2]])
        data_std = np.ones([self.musc_num, self.activation.shape[2]])
        data_trend_u = np.ones([self.musc_num, self.activation.shape[2] - 1])
        data_trend_d = np.ones([self.musc_num, self.activation.shape[2] - 1])
        for i in range(self.musc_num):
            data_mean[i] = np.asarray([np.mean(self.activation[i, :, j]) for j in range(self.activation.shape[2])])
            data_std[i] = np.asarray([np.std(self.activation[i, :, j]) for j in range(self.activation.shape[2])])
            data_trend_u[i] = np.asarray([(np.max(self.activation[i, :, j + 1] - self.activation[i, :, j]) / 0.01)
                                          for j in range(self.activation.shape[2] - 1)])
            data_trend_d[i] = np.asarray([(np.min(self.activation[i, :, j + 1] - self.activation[i, :, j]) / 0.01)
                                          for j in range(self.activation.shape[2] - 1)])
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_trend_u = data_trend_u
        self.data_trend_d = data_trend_d
        np.save(self.folder + 'emg_mean', data_mean)
        np.save(self.folder + 'emg_std', data_std)
        np.save(self.folder + 'emg_trend_u', data_trend_u)
        np.save(self.folder + 'emg_trend_d', data_trend_d)

    def plot_distribution(self):
        num = 0
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if self.musc_num <= 10:
            column_num = 1
        else:
            column_num = 2

        if column_num == 1:
            figure_size = (6, 7.7)
            adjust_params = {'left': 0.125, 'right': 0.9, 'bottom': 0.11, 'top': 0.88,
                             'wspace': 0.2, 'hspace': 0.2}
        else:
            figure_size = (7, 5.7)
            adjust_params = {'left': 0.109, 'right': 0.962, 'bottom': 0.095, 'top': 0.945,
                             'wspace': 0.26, 'hspace': 0.2}

        for figure_num in range(3):
            plt.figure(figsize=figure_size)
            plt.subplots_adjust(**adjust_params)
            for j in range(self.musc_num):
                plt.subplot(math.ceil(self.musc_num / column_num), column_num, j + 1)
                if figure_num == 0:
                    for i in range(self.activation.shape[1]):
                        plt.plot(self.activation[j, i, :], color=color[i % len(color)])
                elif figure_num == 1:
                    plt.errorbar(range(self.data_mean.shape[1]), self.data_mean[j], 2 * self.data_std[j])
                elif figure_num == 2:
                    plt.errorbar(range(self.data_mean.shape[1]), self.data_mean[j], 2 * self.data_std[j],
                                 color='papayawhip')
                    for i in range(self.activation.shape[1]):
                        plt.plot(self.activation[j, i, :], color=color[i % len(color)])

                plt.ylabel(self.parameter.musc_label[j], weight='bold')
                if j < self.musc_num - column_num:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                else:
                    plt.xlabel('timestep', weight='bold')
            num = num + 1
            plt.savefig(self.parameter.result_folder + 'emg_all_{}.png'.format(num))
