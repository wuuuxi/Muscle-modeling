import matplotlib.pyplot as plt
import gekko
import math
from pyomo.environ import *
from pyomo.opt import SolverFactory

from read_files import *


def plot_result(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c, idx=None):
    if include_state == 'lift and down':
        len1 = target_len * 2
        len2 = emg_lift_len * 2
    else:
        len1 = target_len
        len2 = emg_lift_len
    if sport_label == 'deadlift' and joint_idx == 'all':
        emg = emg.T
    t = t[num * len1:(num + 1) * len1]
    r = r[:, num * len1:(num + 1) * len1]
    emg = emg[:, num * len1:(num + 1) * len1]
    time = time[num * len1:(num + 1) * len1]
    torque = torque[num * len1:(num + 1) * len1]
    if len(calu_torque.shape) == 2:
        calu_torque = calu_torque[:, num * len1:(num + 1) * len1]
    else:
        calu_torque = calu_torque[num * len1:(num + 1) * len1]
    emg_std_long = emg_std_long[:, num * len2:(num + 1) * len2]
    emg_mean_long = emg_mean_long[:, num * len2:(num + 1) * len2]
    time_long = resample_by_len(list(time), emg_mean_long.shape[1])
    if idx is None:
        idx = num

    if sport_label == 'biceps_curl':
        if include_TRI is False:
            plt.figure(figsize=(6, 7.7))
            plt.subplot(411)
            plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
            plt.plot(time, torque, label='measured', linewidth=2)
            # plt.xlabel('time (s)')
            plt.ylabel('torque', weight='bold', size=10)
            plt.legend()
            rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
            print("torque rmse", idx, ":\t", "{:.2f}".format(rmse))

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
            # plt.plot(time, np.asarray(emg[0]), label='emg', linewidth=2, zorder=3)
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
            # plt.plot(time, np.asarray(emg[1]), label='emg', linewidth=2, zorder=3)
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
            # plt.plot(time, np.asarray(emg[2]), label='emg', linewidth=2, zorder=3)
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
    elif sport_label == 'bench_press':
        if joint_idx == 'all':
            joint_include = ['elbow', 'shoulder']
            num_column = math.ceil((len(measured_muscle_idx) + 1) / 10)
            if num_column == 1:
                plt.figure(figsize=(3.3, 7.7))
                plt.subplots_adjust(left=0.225, right=0.935)
            else:
                plt.figure(figsize=(1.6 * (num_column + 1), 7.7))
                plt.subplots_adjust(left=0.152, right=0.963, bottom=0.067, top=0.946, wspace=0.403, hspace=0.205)
            for k in range(len(joint_include)):
                plt.subplot(math.ceil((len(measured_muscle_idx) + 1) / num_column), num_column, k + 1)
                plt.plot(time, np.asarray(t[:, k]), label='optimization', linewidth=2)
                plt.plot(time, torque[:, k], label='measured', linewidth=2)
                # plt.plot(time, calu_torque[k], label='calculate', linewidth=2)
                # plt.xlabel('time (s)')
                plt.ylabel(joint_include[k], weight='bold', size=10)
                # plt.legend()
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                rmse = np.sqrt(np.sum((np.asarray(t[:, k]) - torque[:, k]) ** 2) / len(torque[:, k]))
                print("torque rmse", joint_include[k], ":\t", "{:.2f}".format(rmse))
                rmse = np.sqrt(np.sum((calu_torque[k, :] - torque[:, k]) ** 2) / len(torque[:, k]))
                print("torque rmse", joint_include[k], ":\t", "{:.2f}".format(rmse))
            rmse = np.sqrt(np.sum((calu_torque.T - torque) ** 2) / torque.size)
            print("torque rmse:\t", "{:.2f}".format(rmse))
            for j in range(len(measured_muscle_idx)):
                plt.subplot(math.ceil((len(measured_muscle_idx) + 1) / num_column), num_column, j + len(joint_include) + 1)
                plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
                if plot_distribution is True:
                    if mvc_is_variable is True:
                        plt.errorbar(time_long, emg_mean_long[j, :] * c[j], 2 * emg_std_long[j, :] * c[j], label='emg',
                                     color='lavender', zorder=1)
                    else:
                        plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg',
                                     color='lavender', zorder=1)
                else:
                    plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)

                plt.ylabel(musc_label[j], weight='bold')
                if j >= len(measured_muscle_idx) - num_column:
                    plt.xlabel('time (s)')
                else:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                # plt.legend()
        else:
            num_column = math.ceil((len(measured_muscle_idx) + 1) / 10)
            if num_column == 1:
                plt.figure(figsize=(3.3, 7.7))
                plt.subplots_adjust(left=0.225, right=0.935)
            else:
                plt.figure(figsize=(1.6 * (num_column + 1), 7.7))
                plt.subplots_adjust(left=0.152, right=0.963, bottom=0.067, top=0.946, wspace=0.403, hspace=0.205)
            plt.subplot(math.ceil((len(measured_muscle_idx) + 1) / num_column), num_column, 1)
            plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
            plt.plot(time, torque, label='measured', linewidth=2)
            plt.plot(time, calu_torque, label='calculate', linewidth=2)
            plt.ylabel('torque', weight='bold', size=10)
            # plt.xlabel('time (s)')
            # plt.legend()
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
            print("torque rmse", idx, ":\t", "{:.2f}".format(rmse))
            rmse = np.sqrt(np.sum((calu_torque - torque) ** 2) / len(torque))
            print("torque rmse", joint_idx, ":\t", "{:.2f}".format(rmse))
            for j in range(len(measured_muscle_idx)):
                plt.subplot(math.ceil((len(measured_muscle_idx) + 1) / num_column), num_column, j + 2)
                if plot_distribution is True:
                    if mvc_is_variable is True:
                        plt.errorbar(time_long, emg_mean_long[j, :] * c[j], 2 * emg_std_long[j, :] * c[j], label='emg',
                                     color='lavender', zorder=1)
                    else:
                        plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg',
                                     color='lavender',
                                     zorder=1)
                else:
                    plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
                plt.ylabel(musc_label[j], weight='bold')
                if j == len(measured_muscle_idx) - 1:
                    plt.xlabel('time (s)')
                else:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                # plt.legend()
    elif sport_label == 'deadlift':
        if joint_idx == 'all':
            plt.figure(figsize=(8, 7.7))
            plt.subplots_adjust(left=0.090, right=0.990, top=0.970, bottom=0.065, wspace=0.350)
            plot_muscle_num = len(measured_muscle_idx)
            joint_include = ['waist', 'hip', 'knee']
            for k in range(3):
                plt.subplot(math.ceil(plot_muscle_num / 3 + 1), 3, k + 1)
                plt.plot(time, np.asarray(t[:, k]), label='optimization', linewidth=2)
                plt.plot(time, torque[:, k], label='measured', linewidth=2)
                # plt.plot(time, calu_torque[k, :], label='calculate', linewidth=2)
                plt.ylabel(joint_include[k], weight='bold', size=10)
                ax = plt.gca()
                ax.set_xticklabels([])
                # ax.axes.xaxis.set_visible(False)
                rmse = np.sqrt(np.sum((np.asarray(t[:, k]) - torque[:, k]) ** 2) / len(torque[:, k]))
                print("torque rmse", joint_include[k], ":\t", "{:.2f}".format(rmse))
                rmse = np.sqrt(np.sum((calu_torque[k, :] - torque[:, k]) ** 2) / len(torque[:, k]))
                print("torque rmse", joint_include[k], ":\t", "{:.2f}".format(rmse))
            rmse = np.sqrt(np.sum((calu_torque.T - torque) ** 2) / torque.size)
            print("torque rmse:\t", "{:.2f}".format(rmse))
            for j in range(plot_muscle_num):
                plt.subplot(math.ceil(plot_muscle_num / 3 + 1), 3, j + 4)
                plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
                # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                if plot_distribution is True:
                        plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg',
                                     color='lavender', zorder=1)
                else:
                    plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                plt.ylabel(musc_label[j], weight='bold')
                if j >= plot_muscle_num - len(joint_include):
                    plt.xlabel('time (s)')
                else:
                    ax = plt.gca()
                    ax.set_xticklabels([])
                # plt.legend()
        else:
            # plt.figure(figsize=(6, 7.7))
            plt.figure(figsize=(3.3, 7.7))
            plot_muscle_num = len(measured_muscle_idx)
            plt.subplot(plot_muscle_num + 1, 1, 1)
            plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
            plt.plot(time, torque, label='measured', linewidth=2)
            plt.plot(time, calu_torque, label='calculate', linewidth=2)
            plt.subplots_adjust(left=0.225, right=0.935)
            # plt.xlabel('time (s)')
            plt.ylabel('torque', weight='bold', size=10)
            # plt.legend()
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
            print("torque rmse", idx, ":\t", "{:.2f}".format(rmse))
            for j in range(plot_muscle_num):
                plt.subplot(plot_muscle_num + 1, 1, j + 2)
                if plot_distribution is True:
                    if mvc_is_variable is True:
                        plt.errorbar(time_long, emg_mean_long[j, :] * c[j], 2 * emg_std_long[j, :] * c[j], label='emg',
                                     color='lavender', zorder=1)
                    else:
                        plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg', color='lavender',
                                     zorder=1)
                else:
                    plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
                plt.ylabel(musc_label[j], weight='bold')
                if j == plot_muscle_num - 1:
                    plt.xlabel('time (s)')
                else:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                # plt.legend()
            rmse = np.sqrt(np.sum((np.asarray(calu_torque) - torque) ** 2) / len(torque))
            print("torque rmse:\t", "{:.2f}".format(rmse))
    plt.savefig('train_{}.png'.format(idx))


def plot_result_bp(num, t1, t2, r, emg, time, torque1, torque2, emg_std_long, emg_mean_long, calu_torque1, calu_torque2, c, idx=None):
    if include_state == 'lift and down':
        len1 = target_len * 2
        len2 = emg_lift_len * 2
    else:
        len1 = target_len
        len2 = emg_lift_len
    t1 = t1[num * len1:(num + 1) * len1]
    t2 = t2[num * len1:(num + 1) * len1]
    r = r[:, num * len1:(num + 1) * len1]
    emg = emg[:, num * len1:(num + 1) * len1]
    time = time[num * len1:(num + 1) * len1]
    torque1 = torque1[num * len1:(num + 1) * len1]
    torque2 = torque2[num * len1:(num + 1) * len1]
    calu_torque1 = calu_torque1[num * len1:(num + 1) * len1]
    calu_torque2 = calu_torque2[num * len1:(num + 1) * len1]
    emg_std_long = emg_std_long[:, num * len2:(num + 1) * len2]
    emg_mean_long = emg_mean_long[:, num * len2:(num + 1) * len2]
    time_long = resample_by_len(list(time), emg_mean_long.shape[1])
    if idx is None:
        idx = num

    if sport_label == 'bench_press' or sport_label == 'deadlift':
        # plt.figure(figsize=(3.3, 7.7))
        # plt.subplots_adjust(left=0.225, right=0.935)
        if num == 0:
            plt.figure(figsize=(2.5, 6.7))
            plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
        else:
            plt.figure(figsize=(2.3, 6.7))
            plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)
        plt.subplot(8, 1, 1)
        plt.plot(time, np.asarray(t1), label='optimization', linewidth=2)
        plt.plot(time, torque1, label='measured', linewidth=2)
        # plt.xlabel('time (s)')
        if legend_label is True or num == 0:
            plt.ylabel('torque1', weight='bold', size=10)
        # plt.legend()
        ax = plt.gca()
        ax.set_xticklabels([])

        plt.subplot(8, 1, 2)
        plt.plot(time, np.asarray(t2), label='optimization', linewidth=2)
        plt.plot(time, torque2, label='measured', linewidth=2)
        if legend_label is True or num == 0:
            plt.ylabel('torque2', weight='bold', size=10)
        # plt.legend()
        ax = plt.gca()
        ax.set_xticklabels([])

        for j in range(len(measured_muscle_idx)):
            plt.subplot(8, 1, j + 3)
            if plot_distribution is True:
                if mvc_is_variable is True:
                    plt.errorbar(time_long, emg_mean_long[j, :] * c[j], 2 * emg_std_long[j, :] * c[j], label='emg',
                                 color='lavender', zorder=1)
                else:
                    plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg', color='lavender',
                                 zorder=1)
            else:
                plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
            # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)  # zhushi
            plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
            if legend_label is True or num == 0:
                plt.ylabel(musc_label[j], weight='bold')
            if j == len(measured_muscle_idx) - 1:
                plt.xlabel('time (s)')
            else:
                ax = plt.gca()
                ax.set_xticklabels([])
            # plt.legend()
    plt.savefig('train_{}.png'.format(idx))

    rmse1 = np.sqrt(np.sum((np.asarray(calu_torque1) - torque1) ** 2) / len(torque1))
    rmse2 = np.sqrt(np.sum((np.asarray(calu_torque2) - torque2) ** 2) / len(torque2))
    print("calu torque rmse:\t", "{:.2f}".format(rmse1))
    print("calu torque rmse:\t", "{:.2f}".format(rmse2))


def plot_result_dl(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c, idx=None):
    torque1 = torque[:, 0]
    torque2 = torque[:, 1]
    torque3 = torque[:, 2]
    calu_torque1 = calu_torque[0]
    calu_torque2 = calu_torque[1]
    calu_torque3 = calu_torque[2]
    if include_state == 'lift and down':
        len1 = target_len * 2
        len2 = emg_lift_len * 2
    else:
        len1 = target_len
        len2 = emg_lift_len
    t1 = t[num * len1:(num + 1) * len1, 0]
    t2 = t[num * len1:(num + 1) * len1, 1]
    t3 = t[num * len1:(num + 1) * len1, 2]
    r = r[:, num * len1:(num + 1) * len1]
    emg = emg[:, num * len1:(num + 1) * len1]
    time = time[num * len1:(num + 1) * len1]
    torque1 = torque1[num * len1:(num + 1) * len1]
    torque2 = torque2[num * len1:(num + 1) * len1]
    torque3 = torque3[num * len1:(num + 1) * len1]
    calu_torque1 = calu_torque1[num * len1:(num + 1) * len1]
    calu_torque2 = calu_torque2[num * len1:(num + 1) * len1]
    calu_torque3 = calu_torque3[num * len1:(num + 1) * len1]
    emg_std_long = emg_std_long[:, num * len2:(num + 1) * len2]
    emg_mean_long = emg_mean_long[:, num * len2:(num + 1) * len2]
    time_long = resample_by_len(list(time), emg_mean_long.shape[1])
    if idx is None:
        idx = num

    if sport_label == 'bench_press' or sport_label == 'deadlift':
        # plt.figure(figsize=(3.3, 7.7))
        # plt.subplots_adjust(left=0.225, right=0.935)
        if num == 0:
            plt.figure(figsize=(2.5, 6.7))
            plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
        else:
            plt.figure(figsize=(2.3, 6.7))
            plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)
        plt.subplot(9, 1, 1)
        plt.plot(time, np.asarray(t1), label='optimization', linewidth=2)
        plt.plot(time, torque1, label='measured', linewidth=2)
        # plt.xlabel('time (s)')
        if legend_label is True or num == 0:
            plt.ylabel('torque1', weight='bold', size=10)
        # plt.legend()
        ax = plt.gca()
        ax.set_xticklabels([])

        plt.subplot(9, 1, 2)
        plt.plot(time, np.asarray(t2), label='optimization', linewidth=2)
        plt.plot(time, torque2, label='measured', linewidth=2)
        if legend_label is True or num == 0:
            plt.ylabel('torque2', weight='bold', size=10)
        # plt.legend()
        ax = plt.gca()
        ax.set_xticklabels([])

        plt.subplot(9, 1, 3)
        plt.plot(time, np.asarray(t3), label='optimization', linewidth=2)
        plt.plot(time, torque3, label='measured', linewidth=2)
        if legend_label is True or num == 0:
            plt.ylabel('torque3', weight='bold', size=10)
        # plt.legend()
        ax = plt.gca()
        ax.set_xticklabels([])

        for j in range(6):
            plt.subplot(9, 1, j + 4)
            if plot_distribution is True:
                if mvc_is_variable is True:
                    plt.errorbar(time_long, emg_mean_long[j, :] * c[j], 2 * emg_std_long[j, :] * c[j], label='emg',
                                 color='lavender', zorder=1)
                else:
                    plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg', color='lavender',
                                 zorder=1)
            else:
                plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
            # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)  # zhushi
            plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
            if legend_label is True or num == 0:
                plt.ylabel(musc_label[j], weight='bold')
            if j == 5:
                plt.xlabel('time (s)')
            else:
                ax = plt.gca()
                ax.set_xticklabels([])
            # plt.legend()
    plt.savefig('train_{}.png'.format(idx))


def plot_all_result(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c=0):
    # rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
    # print("torque rmse:\t", "{:.2f}".format(rmse))
    for i in range(num):
        plot_result(i, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)


def plot_all_result_bp(num, t1, t2, r, emg, time, torque1, torque2, emg_std_long, emg_mean_long, calu_torque1, calu_torque2, c=0):
    rmse = np.sqrt(np.sum((np.asarray(t1) - torque1) ** 2 + (np.asarray(t2) - torque2) ** 2) / len(torque1 + torque2))
    rmse1 = np.sqrt(np.sum((np.asarray(t1) - torque1) ** 2) / len(torque1))
    rmse2 = np.sqrt(np.sum((np.asarray(t2) - torque2) ** 2) / len(torque2))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    print("torque rmse1:\t", "{:.2f}".format(rmse1))
    print("torque rmse2:\t", "{:.2f}".format(rmse2))
    for i in range(num):
        plot_result_bp(i, t1, t2, r, emg, time, torque1, torque2, emg_std_long, emg_mean_long, calu_torque1, calu_torque2, c)


def plot_all_result_dl(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c=0):
    rmse = np.sqrt(np.sum((np.asarray(t[:, 0]) - torque[:, 0]) ** 2
                          + (np.asarray(t[:, 1]) - torque[:, 1]) ** 2
                          + (np.asarray(t[:, 2]) - torque[:, 2]) ** 2) / len(torque[:, 0] + torque[:, 1]))
    rmse1 = np.sqrt(np.sum((np.asarray(t[:, 0]) - torque[:, 0]) ** 2) / len(torque[:, 0]))
    rmse2 = np.sqrt(np.sum((np.asarray(t[:, 1]) - torque[:, 1]) ** 2) / len(torque[:, 1]))
    rmse3 = np.sqrt(np.sum((np.asarray(t[:, 2]) - torque[:, 2]) ** 2) / len(torque[:, 2]))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    print("torque rmse1:\t", "{:.2f}".format(rmse1))
    print("torque rmse2:\t", "{:.2f}".format(rmse2))
    print("torque rmse3:\t", "{:.2f}".format(rmse3))
    for i in range(num):
        plot_result_dl(i, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)


def plot_all_result_squat(num, t, a, torque, emg, time, mean, std, idx=None):
    for i in range(num):
        plot_result_squat(i, t, a, torque, emg, time, mean, std, idx)


def plot_all_result_one(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c):
    rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    for i in range(num):
        plot_result(i, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)


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
            # if j < arm.shape[1] - 1:
            #     m.Equation(x[i][j + 1] <= x[i][j] + target_len / emg_lift_len * 5 * emg_trend_u[i][j])
            #     m.Equation(x[i][j + 1] >= x[i][j] + target_len / emg_lift_len * 5 * emg_trend_d[i][j])
            #     # m.Equation(x[i][j + 1] <= x[i][j] + target_len / emg_lift_len * 10 * emg_trend_u[i][j])
            #     # m.Equation(x[i][j + 1] >= x[i][j] + target_len / emg_lift_len * 10 * emg_trend_d[i][j])
            m.Equation(x[i][j] <= (emg_mean[i][j] + emg_std[i][j] * 2.5))
            m.Equation(x[i][j] >= (emg_mean[i][j] - emg_std[i][j] * 2.5))
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
    plt.ylabel('biceps', weight='bold')
    rmse = np.sqrt(np.sum((act[0, :] - emg[0, :]) ** 2) / len(emg[0, :]))
    print("torque rmse", num, ":\t", "{:.4f}".format(rmse))
    plt.legend()

    plt.subplot(413)
    plt.errorbar(time_long, emg_mean_long[1, :], 2 * emg_std_long[1, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(act[1, :]), label='optimization', linewidth=2, zorder=2)
    plt.plot(time, np.asarray(emg[1, :]), label='emg', linewidth=2, zorder=3)
    # plt.xlabel('time (s)')
    plt.ylabel('brachialis', weight='bold')
    rmse = np.sqrt(np.sum((act[1, :] - emg[1, :]) ** 2) / len(emg[1, :]))
    print("torque rmse", num, ":\t", "{:.4f}".format(rmse))
    plt.legend()

    plt.subplot(414)
    plt.errorbar(time_long, emg_mean_long[2, :], 2 * emg_std_long[2, :], label='emg', color='lavender', zorder=1)
    plt.plot(time, np.asarray(act[2, :]), label='optimization', linewidth=2, zorder=2)
    plt.plot(time, np.asarray(emg[2, :]), label='emg', linewidth=2, zorder=3)
    plt.xlabel('time (s)', weight='bold', size=10)
    plt.ylabel('brachiorad', weight='bold')
    rmse = np.sqrt(np.sum((act[2, :] - emg[2, :]) ** 2) / len(emg[2, :]))
    print("torque rmse", num, ":\t", "{:.4f}".format(rmse))
    plt.legend()
    plt.savefig('test_{}.png'.format(num))


def test_optimization_emg_bp(num, y, torque1, torque2, time, emg_mean_long, emg_std_long, arm1, arm2, emg, emg_mean, emg_std, emg_trend_u, emg_trend_d):
    time_long = resample_by_len(list(time), emg_mean_long.shape[1])

    shape0 = arm1.shape[0]
    shape1 = arm1.shape[1]

    # 创建一个具体的模型
    model = ConcreteModel()
    model.I = RangeSet(1, shape0)
    model.J = RangeSet(1, shape1)

    # 定义变量
    model.x = Var(model.I, model.J, within=NonNegativeReals)  # activation
    model.c = Var(model.I, within=NonNegativeReals)  # MVC emg
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t1 = Var(model.J, within=Reals)  # torque
    model.t2 = Var(model.J, within=Reals)  # torque

    # 定义约束条件
    model.constr = ConstraintList()
    for j in model.J:
        for i in model.I:
            # model.constr.add(model.x[i, j + 1] >= model.x[i, j] + target_len/emg_lift_len*5*emg_trend_d[i - 1][j - 1])
            # model.constr.add(model.x[i, j + 1] <= model.x[i, j] + target_len/emg_lift_len*5*emg_trend_u[i - 1][j - 1])
            model.constr.add(model.x[i, j] >= emg_mean[i - 1, j - 1] - emg_std[i - 1, j - 1] * 2.5)
            model.constr.add(model.x[i, j] <= emg_mean[i - 1, j - 1] + emg_std[i - 1, j - 1] * 2.5)
            # model.constr.add(model.x[i, j] >= emg_mean[i - 1, j - 1] - emg_std[i - 1, j - 1] * 4)
            # model.constr.add(model.x[i, j] <= emg_mean[i - 1, j - 1] + emg_std[i - 1, j - 1] * 4)
            model.constr.add(model.x[i, j] >= 0)
            model.constr.add(model.x[i, j] <= 1)
            model.constr.add(model.f[i, j] == model.x[i, j] * y[i - 1])  # muscle force
        model.constr.add(model.t1[j] == sum(model.f[i, j] * arm1[i - 1, j - 1] for i in model.I))  # torque1`
        model.constr.add(model.t2[j] == sum(model.f[i, j] * arm2[i - 1, j - 1] for i in model.I))  # torque2`

    # 定义目标函数
    # obj = sum(np.square(model.t[j] - torque[j - 1]) for j in model.J)
    obj = sum((model.t1[j] - torque1[j - 1]) ** 2 for j in model.J) / shape1 + \
          sum((model.t2[j] - torque2[j - 1]) ** 2 for j in model.J) / shape1
    model.obj = Objective(expr=obj, sense=minimize)

    # 求解器配置
    solver = SolverFactory('ipopt')

    # 创建一个结果列表来保存迭代过程中的目标函数值
    results = []

    # 求解优化问题，并记录迭代过程中的目标函数值
    def solve_optimization():
        result = solver.solve(model)
        results.append(value(model.obj))
        return result

    # # 迭代求解优化问题，直到收敛
    # while True:
    #     result = solve_optimization()
    #     if result.solver.termination_condition == TerminationCondition.optimal:
    #         break

    # 求解优化问题
    results = solver.solve(model)

    c = np.ones_like(arm1)
    e1 = np.ones_like(torque1)
    e2 = np.ones_like(torque2)
    for i in model.I:
        for j in model.J:
            c[i - 1, j - 1] = value(model.x[i, j])
    for j in model.J:
        e1[j - 1] = value(model.t1[j])
        e2[j - 1] = value(model.t2[j])
    act = c
    t1 = e1
    t2 = e2
    tor1 = np.asarray(t1)
    tor2 = np.asarray(t2)

    # plt.figure(figsize=(3.3, 7.7))
    # plt.subplots_adjust(left=0.225, right=0.935)
    plt.figure(figsize=(2.3, 6.7))
    plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)
    # plt.figure(figsize=(2.5, 6.7))
    # plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    plt.subplot(811)
    plt.plot(time, tor1, label='optimization', linewidth=2)
    plt.plot(time, torque1, label='measured', linewidth=2)
    if legend_label is True:
        plt.ylabel('torque1', weight='bold', size=10)
    # plt.legend()
    ax = plt.gca()
    ax.set_xticklabels([])
    rmse = np.sqrt(np.sum((tor1 - torque1) ** 2) / len(torque1))
    print("torque rmse", ":\t", "{:.2f}".format(rmse))

    plt.subplot(812)
    plt.plot(time, tor2, label='optimization', linewidth=2)
    plt.plot(time, torque2, label='measured', linewidth=2)
    if legend_label is True:
        plt.ylabel('torque2', weight='bold', size=10)
    # plt.legend()
    ax = plt.gca()
    ax.set_xticklabels([])
    rmse = np.sqrt(np.sum((tor2 - torque2) ** 2) / len(torque2))
    print("torque rmse", ":\t", "{:.2f}".format(rmse))

    for j in range(len(measured_muscle_idx)):
        plt.subplot(8, 1, j + 3)
        plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg', color='lavender', zorder=1)
        plt.plot(time, np.asarray(act[j, :]), label='optimization', linewidth=2, zorder=2)
        plt.plot(time, np.asarray(emg[j, :]), label='emg', linewidth=2, zorder=3)
        if legend_label is True:
            plt.ylabel(musc_label[j], weight='bold')
        rmse = np.sqrt(np.sum((act[j, :] - emg[j, :]) ** 2) / len(emg[0, :]))
        print("torque rmse", num, ":\t", "{:.4f}".format(rmse))
        # plt.legend()
        if j != len(measured_muscle_idx) - 1:
            ax = plt.gca()
            ax.set_xticklabels([])
        else:
            plt.xlabel('time (s)')
    plt.savefig('test_{}.png'.format(num))

    # plt.figure()
    # np.save('predication1', act)
    # # np.save('time', time)
    # plt.plot(time, np.asarray(act[1, :]))
    # plt.title('Triceps Brachii', weight='bold')
    # plt.ylabel('Activation', weight='bold')
    # plt.xlabel('Time(s)', weight='bold')


def test_optimization_emg_dl(num, y, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std, trend_u, trend_d):
    time_long = resample_by_len(list(time), emg_mean_long.shape[1])

    shape0 = arm.shape[0]
    shape1 = arm.shape[1]
    shape2 = arm.shape[2]

    # 创建一个具体的模型
    model = ConcreteModel()
    model.I = RangeSet(0, shape0 - 1)
    model.J = RangeSet(0, shape1 - 1)
    model.K = RangeSet(0, shape2 - 1)

    # 定义变量
    model.x = Var(model.I, model.J, within=NonNegativeReals)  # activation
    # model.c = Var(model.I, within=NonNegativeReals)  # MVC emg
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t = Var(model.J, model.K, within=Reals)  # torque

    # 定义约束条件
    model.constr = ConstraintList()
    for j in model.J:
        for i in model.I:
            # model.constr.add(model.x[i, j + 1] >= model.x[i, j] + target_len/emg_lift_len*5*trend_d[i][j])
            # model.constr.add(model.x[i, j + 1] <= model.x[i, j] + target_len/emg_lift_len*5*trend_u[i][j])
            model.constr.add(model.x[i, j] >= emg_mean[i, j] - emg_std[i, j] * 2.5)
            model.constr.add(model.x[i, j] <= emg_mean[i, j] + emg_std[i, j] * 2.5)
            model.constr.add(model.x[i, j] >= 0)
            model.constr.add(model.x[i, j] <= 1)
            model.constr.add(model.f[i, j] == model.x[i, j] * y[i])  # muscle force
        for k in range(shape2):
            model.constr.add(model.t[j, k] == sum(arm[i, j, k] * model.f[i, j] for i in model.I))  # torque

    # objective function
    obj = sum((sum((model.t[j, k] - torque[j, k]) ** 2 for j in model.J) / shape1) for k in model.K)
    model.obj = Objective(expr=obj, sense=minimize)

    # 求解器
    solver = SolverFactory('ipopt')
    solver.solve(model)

    c = np.ones_like(emg)
    e = np.ones_like(torque)
    for i in model.I:
        for j in model.J:
            c[i, j] = value(model.x[i, j])
    for j in model.J:
        for k in model.K:
            e[j, k] = value(model.t[j, k])
    act = np.asarray(c)
    tor = np.asarray(e)

    plot_muscle_num = len(measured_muscle_idx)
    num_column = math.ceil((plot_muscle_num + 1) / 10)
    if sport_label == 'bench_press':
        joint_include = ['elbow', 'shoulder']
    elif sport_label == 'deadlift':
        joint_include = ['waist', 'hip', 'knee']

    if num_column == 1:
        plt.figure(figsize=(3.3, 7.7))
        plt.subplots_adjust(left=0.225, right=0.935)
    elif num_column == 2:
        plt.figure(figsize=(4.8, 7.7))
        plt.subplots_adjust(left=0.152, right=0.963, bottom=0.067, top=0.946, wspace=0.403, hspace=0.205)
    elif num_column == 3:
        plt.figure(figsize=(8, 7.7))
        plt.subplots_adjust(left=0.090, right=0.990, bottom=0.065, top=0.970, wspace=0.350)
    else:
        plt.figure(figsize=(1.6 * (num_column + 1), 7.7))
        plt.subplots_adjust(left=0.152, right=0.963, bottom=0.067, top=0.946, wspace=0.403, hspace=0.205)

    for k in range(len(joint_include)):
        plt.subplot(math.ceil(plot_muscle_num / num_column + 1), num_column, k + 1)
        plt.plot(time, tor[:, k], label='optimization', linewidth=2)
        plt.plot(time, torque[:, k], label='measured', linewidth=2)
        # plt.plot(time, calu_torque[k, :], label='calculate', linewidth=2)
        if legend_label is True:
            plt.ylabel(joint_include[k], weight='bold', size=10)
        ax = plt.gca()
        ax.set_xticklabels([])
        # ax.axes.xaxis.set_visible(False)
        rmse = np.sqrt(np.sum((tor[:, k] - torque[:, k]) ** 2) / len(torque[:, k]))
        print("torque rmse", joint_include[k], ":\t", "{:.2f}".format(rmse))
    rmse = np.sqrt(np.sum((tor - torque) ** 2) / torque.size)
    print("torque rmse:\t", "{:.2f}".format(rmse))
    rmse_emg = np.sqrt(np.sum((act - emg) ** 2) / emg.size)
    print("emg rmse:\t", "{:.2f}".format(rmse_emg))
    for j in range(plot_muscle_num):
        plt.subplot(math.ceil(plot_muscle_num / num_column + 1), num_column, j + num_column + 1)
        plt.errorbar(time_long, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg', color='lavender', zorder=1)
        plt.plot(time, act[j, :], label='optimization', linewidth=2, zorder=2, color='#1f77b4')
        plt.plot(time, emg[j, :], label='emg', linewidth=2, zorder=3, color='#ff7f0e')
        plt.ylabel(musc_label[j], weight='bold')
        if j >= plot_muscle_num - len(joint_include):
            plt.xlabel('time (s)')
        else:
            ax = plt.gca()
            ax.set_xticklabels([])
        # plt.legend()
    plt.savefig('test_{}.png'.format(num))


def application(label, fmax, idx):
    if idx is None:
        if label == 'zhuo-left-1kg':
            idx = ['3', '4', '6', '7', '9']
        elif label == 'chenzui-left-all-2.5kg':
            idx = ['5', '6', '7', '8', '9']
        elif label == 'chenzui-left-all-3kg':
            idx = ['10', '11', '12', '13', '14']
        elif label == 'chenzui-left-all-4kg':
            idx = ['15', '16', '17']
        elif label == 'chenzui-left-all-6.5kg':
            idx = ['18', '19', '20', '21', '22']
        elif label == 'bp-chenzui-left-4kg':
            idx = ['1', '2', '3', '4', '5', '6']
        # elif label == 'bp-chenzui-left-5.5kg' or label == 'bp-chenzui-left-6.5kg':
        #     idx = ['6', '7', '8']
        elif label == 'bp-chenzui-left-5.5kg':
            idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif label == 'bp-chenzui-left-6.5kg':
            idx = ['1', '2', '3', '4', '5', '6', '7', '8']
        elif label == 'bp-chenzui-left-7kg':
            idx = ['1', '2', '3', '4', '5', '6']
        elif label == 'bp-chenzui-left-9.5kg':
            idx = ['1', '2', '3', '4', '5', '6', '7']
        elif label == 'bp-zhuo-right-3kg':
            idx = ['4', '5', '6']
        elif label == 'bp-zhuo-right-4kg':
            idx = ['1', '2', '3', '4', '5', '6']
        elif label == 'bp-yuetian-right-20kg':
            idx = ['1', '2', '3', '4', '5', '6']
        elif label == 'bp-yuetian-right-30kg':
            idx = ['1', '2', '3', '4', '5', '6']
        elif label == 'bp-yuetian-right-40kg':
            idx = ['1', '2', '3', '4', '5', '6']
        elif label == 'bp-yuetian-right-50kg':
            idx = ['1', '2', '3', '4', '5', '6']
        elif label == 'bp-yuetian-right-60kg':
            idx = ['1', '2', '3', '4', '5', '6']
    if sport_label == 'biceps_curl':
        for i in range(len(idx)):
            emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
                = read_realted_files(label=label, idx=idx[i])
            test_optimization_emg(i, fmax, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std,
                                  emg_trend_u, emg_trend_d)
    elif sport_label == 'bench_press':
        if label == 'bp-yuetian-right-all':
            labels = ['bp-yuetian-right-20kg', 'bp-yuetian-right-50kg']
            idxs = [['2', '3', '4'],
                    ['2', '3', '4']]
            for k in range(len(labels)):
                label = labels[k]
                idx = idxs[k]
                print(label)
                for i in range(len(idx)):
                    o = read_groups_files(label, idx[i])
                    test_optimization_emg_dl(i, fmax, o['torque'], o['time'], o['emg_mean_long'], o['emg_std_long'],
                                             o['arm'], o['emg'], o['emg_mean'], o['emg_std'], o['trend_u'],
                                             o['trend_d'])
        else:
            for i in range(len(idx)):
                o = read_realted_files_bp(label=label, idx=idx[i])
                test_optimization_emg_dl(i, fmax, o['torque'], o['time'], o['emg_mean_long'], o['emg_std_long'],
                                         o['arm'], o['emg'], o['emg_mean'], o['emg_std'], o['trend_u'], o['trend_d'])
    elif sport_label == 'deadlift':
        if label == 'dl-yuetian-right-all':
            labels = ['dl-yuetian-right-35kg', 'dl-yuetian-right-45kg', 'dl-yuetian-right-65kg', 'dl-yuetian-right-75kg']
            idxs = [['2', '3', '5', '6'],
                    ['1', '2', '3', '5'],
                    ['1', '2', '3', '4', '5'],
                    ['1', '2', '3', '4', '5', '6']]
            for k in range(len(labels)):
                label = labels[k]
                idx = idxs[k]
                print(label)
                for i in range(len(idx)):
                    o = read_groups_files(label, idx[i])
                    test_optimization_emg_dl(i, fmax, o['torque'], o['time'], o['emg_mean_long'], o['emg_std_long'],
                                             o['arm'], o['emg'], o['emg_mean'], o['emg_std'], o['trend_u'],
                                             o['trend_d'])
        else:
            for i in range(len(idx)):
                o = read_groups_files(label, idx[i])
                test_optimization_emg_dl(i, fmax, o['torque'], o['time'], o['emg_mean_long'], o['emg_std_long'],
                                         o['arm'], o['emg'], o['emg_mean'], o['emg_std'], o['trend_u'], o['trend_d'])


if __name__ == '__main__':
    y_r = np.array([436.65, 157.00, 15])
    # y_r = np.array([134.57, 846.82, 150.00])
    # y_r = np.array([8126.39, 15700.00, 485.59])
    # y_r = np.array([1484.95, 242.86, 445.27])

    emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
        = read_groups_files('chenzui-left-all-6.5kg')

    plot_all_result(5, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)

    test_result(c_r, y_r)
    application('zhuo-left-1kg', ['3', '4', '6', '7', '9'], y_r)
    plt.show()
