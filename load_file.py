import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import sys
from basic import *


class MuscleOneGroup:
    def __init__(self, parameter, moment_file, moment_arm_file, joints, emg, weight):
        self.parameter = parameter
        self.moment_file = moment_file
        self.moment_arm_file = moment_arm_file
        self.joints = joints
        self.emg = emg
        self.weight = weight

        self.rep_num = self.emg.rep_num
        self.joint_num = len(self.joints)
        self._unified_len = self.parameter.opti_length
        self.idx_list = self.emg.idx_list

        self.torque_long = None
        self.arm_long = None
        self.torque = None
        self.arm = None
        self.time = None
        self.activation = None
        self.emg_mean = None
        self.emg_std = None
        self.emg_trend_d = None
        self.emg_trend_u = None
        self.data_preprocessing()

    def data_preprocessing(self):
        time_torque = self.moment_file['time']
        time_moment_arm = self.moment_arm_file[0]['time']
        torque = []
        moment_arm_long = [[] for _ in range(self.joint_num)]
        for joint in self.joints:
            torque_filtered = scipy.signal.savgol_filter(self.moment_file[joint], 53, 3)
            torque.append(torque_filtered)
        self.torque_long = torque
        for m in self.parameter.musc_opensim:
            for i in range(self.joint_num):
                moment_arm_long[i].append(self.moment_arm_file[i][m])
        self.arm_long = moment_arm_long

        tor = []
        arm = [[] for _ in range(self.joint_num)]
        time = []
        for idx in self.idx_list - 1:
            time_tor, tor_rep = self.time_alignment(time_torque, torque, self.emg.timestep[idx])
            tor.append(tor_rep)
            time.append(time_tor)
            for i in range(self.joint_num):
                time_arm, arm_rep = self.time_alignment(time_moment_arm, moment_arm_long[i], self.emg.timestep[idx])
                arm[i].append(arm_rep)
        self.torque = np.asarray(tor)
        self.arm = np.asarray(arm)
        self.time = np.asarray(time)

        emg_mean = []
        emg_std = []
        emg_trend_d = []
        emg_trend_u = []
        activation = [[] for _ in range(self.rep_num)]
        for i in range(self.emg.musc_num):
            emg_mean.append(resample_by_len(self.emg.data_mean[i], self._unified_len * 2))
            emg_std.append(resample_by_len(self.emg.data_std[i], self._unified_len * 2))
            emg_trend_d.append(resample_by_len(self.emg.data_trend_d[i], self._unified_len * 2))
            emg_trend_u.append(resample_by_len(self.emg.data_trend_u[i], self._unified_len * 2))
            for j in range(self.rep_num):
                activation[j].append(resample_by_len(self.emg.activation[i, j], self._unified_len * 2))
        self.emg_mean = np.asarray(emg_mean)
        self.emg_std = np.asarray(emg_std)
        self.emg_trend_d = np.asarray(emg_trend_d)
        self.emg_trend_u = np.asarray(emg_trend_u)
        self.activation = np.asarray(activation)

    def time_alignment(self, data_time, data, timestep):
        data = np.asarray(data)
        t_step = np.zeros(4, dtype=int)
        data_rep = []
        for i in range(4):
            t_step[i] = find_nearest_idx(data_time, timestep[i])

        def process_data_slice(data_slice):
            return (resample_by_len(list(data_slice[t_step[0]:t_step[1]]), self._unified_len) +
                    resample_by_len(list(data_slice[t_step[2]:t_step[3]]), self._unified_len))

        time = process_data_slice(data_time)
        for idx in np.ndindex(data.shape[:-1]):
            data_rep.append(process_data_slice(data[idx]))
        return time, data_rep


class MuscleData:
    def __init__(self, emg, json_data, parameter):
        self.emg = emg
        self.json_data = json_data
        self.parameter = parameter
        self.folder = json_data['folder']
        self.weights = json_data["weights"]
        self.weights_num = len(self.weights)
        assert len(json_data["MA"]) == len(json_data["joint"])
        print("\r", end="")
        print("Load file: 0%: ", "▋" * (0 * 10), end="")
        sys.stdout.flush()

        self.muscle_groups = []
        for idx in range(self.weights_num):
            weight = self.weights[idx]
            emg_idx = self.emg.label.index(weight)
            # print('-' * 25, weight, 'kg', '-' * 25)
            moment_arm = []
            for joint in json_data["MA"]:
                moment_arm.append(pd.read_excel(self.folder + joint + weight + '.xlsx'))
            moment = pd.read_excel(self.folder + json_data["ID"] + weight + '.xlsx')
            file_record = MuscleOneGroup(parameter, moment, moment_arm, json_data["joint"],
                                         self.emg.emg_groups[emg_idx], weight)
            self.muscle_groups.append(file_record)

            print("\r", end="")
            print("Load basic data:", '\t', " {}%: ".format((idx + 1) / self.weights_num * 100),
                  "▋" * ((idx + 1) * int(30 / self.weights_num)), end="")
            sys.stdout.flush()
            time.sleep(0.05)
        print("\n", end="")
