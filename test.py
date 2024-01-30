import matplotlib.pyplot as plt
import gekko

from read_files import *


def plot_result(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c, idx=None):
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
    if idx is None:
        idx = num
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
    plt.savefig('train_{}.png'.format(idx))

    # plt.figure(figsize=(6, 3.7))
    # plt.plot(time, calu_torque, label='calculated', linewidth=2)
    # plt.plot(time, torque, label='measured', linewidth=2)
    # plt.xlabel('time (s)', weight='bold')
    # plt.ylabel('torque', weight='bold')
    # plt.legend()
    rmse = np.sqrt(np.sum((np.asarray(calu_torque) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))


def plot_all_result(num, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c):
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


def application(label, y_r):
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
    else:
        print('No such label!')
        return 0
    for i in range(len(idx)):
        emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
            = read_realted_files(label=label, idx=idx[i])
        test_optimization_emg(i, y_r, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std,
                              emg_trend_u, emg_trend_d)
    # emg_mean1, emg_std1, arm1, torque1, time1, emg_mean_long1, emg_std_long1, emg1, time_emg1, emg_trend_u1, emg_trend_d1 \
    #     = read_realted_files(label=label, idx=idx[0])
    # emg_mean2, emg_std2, arm2, torque2, time2, emg_mean_long2, emg_std_long2, emg2, time_emg2, emg_trend_u2, emg_trend_d2 \
    #     = read_realted_files(label=label, idx=idx[1])
    # emg_mean3, emg_std3, arm3, torque3, time3, emg_mean_long3, emg_std_long3, emg3, time_emg3, emg_trend_u3, emg_trend_d3 \
    #     = read_realted_files(label=label, idx=idx[2])
    # emg_mean4, emg_std4, arm4, torque4, time4, emg_mean_long4, emg_std_long4, emg4, time_emg4, emg_trend_u4, emg_trend_d4 \
    #     = read_realted_files(label=label, idx=idx[3])
    # emg_mean5, emg_std5, arm5, torque5, time5, emg_mean_long5, emg_std_long5, emg5, time_emg5, emg_trend_u5, emg_trend_d5 \
    #     = read_realted_files(label=label, idx=idx[4])
    # # emg_mean6, emg_std6, arm6, fa6, torque6, time6, emg_mean_long6, emg_std_long6, emg6, time_emg6 \
    # #     = read_chenzui_realted_files(label='zhuo-9')
    # test_optimization_emg(0, y_r, torque1, time1, emg_mean_long1, emg_std_long1, arm1, emg1, emg_mean1, emg_std1,
    #                       emg_trend_u1, emg_trend_d1)
    # test_optimization_emg(1, y_r, torque2, time2, emg_mean_long2, emg_std_long2, arm2, emg2, emg_mean2, emg_std2,
    #                       emg_trend_u2, emg_trend_d2)
    # test_optimization_emg(2, y_r, torque3, time3, emg_mean_long3, emg_std_long3, arm3, emg3, emg_mean3, emg_std3,
    #                       emg_trend_u3, emg_trend_d3)
    # test_optimization_emg(3, y_r, torque4, time4, emg_mean_long4, emg_std_long4, arm4, emg4, emg_mean4, emg_std4,
    #                       emg_trend_u4, emg_trend_d4)
    # test_optimization_emg(4, y_r, torque5, time5, emg_mean_long5, emg_std_long5, arm5, emg5, emg_mean5, emg_std5,
    #                       emg_trend_u5, emg_trend_d5)
    # # test_optimization_emg(5, y_r, torque6, time6, emg_mean_long6, emg_std_long6, arm6, emg6, emg_mean6, emg_std6,
    # #                       emg_trend_u6, emg_trend_d6)


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
