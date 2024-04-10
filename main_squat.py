import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy
from scipy import signal
from scipy.optimize import minimize
from pyomo.environ import *
from pyomo.opt import SolverFactory

# from emg_distribution import *
# from test import *
# from require import *


EMG1Group = ['tibant_l', 'edl_l']
EMG2Group = ['gaslat_l', 'gasmed_l', 'soleus_l']
EMG3Group = ['vaslat_l', 'vasint_l', 'vasmed_l']
EMG4Group = ['recfem_l']
EMG5Group = ['addbrev_l', 'addlong_l', 'addmagDist_l', 'addmagIsch_l', 'addmagMid_l', 'addmagProx_l']
EMG6Group = ['bflh_l', 'bfsh_l', 'semimem_l', 'semiten_l']
EMG6bGroup = ['glmax1_l', 'glmax2_l', 'glmax3_l', 'glmed1_l', 'glmed2_l', 'glmed3_l', 'glmin1_l', 'glmin2_l',
              'glmin3_l', 'tfl_l']
EMG7Group = ['tibant_r', 'edl_r']
EMG8Group = ['gaslat_r', 'gasmed_r', 'soleus_r']
EMG9Group = ['vaslat_r', 'vasint_r', 'vasmed_r']
EMG10Group = ['recfem_r']
EMG11Group = ['addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 'addmagMid_r', 'addmagProx_r']
EMG12Group = ['bflh_r', 'bfsh_r', 'semimem_r', 'semiten_r']

# TorqueList = [10, 11, 12, 13]
TorqueList = [10, 12, 13]
TorqueList_t = [9, 20, 29]
muscle_idx = [0, 2, 5, 8, 9, 15]
TarMuscleList = [EMG1Group, EMG2Group, EMG3Group, EMG4Group, EMG5Group, EMG6Group]
total_muscle_num = 19
musc_label = ['tibant', 'gaslat', 'vaslat', 'recfem', 'addbrev', 'bflh']

length = 500
folder = 'files/squat/output/trials/'
plot_distribution = True


def resample_by_len(orig_list: list, target_len: int):
    orig_list = np.asarray(orig_list)
    if len(orig_list.shape) == 1:
        orig_list_len = orig_list.shape[0]
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
        f = interpolate.interp1d(x, orig_list[:, 0], 'linear')
        del x
        resample_list = f([x for x in range(target_len)])
        resample_list = np.asarray(resample_list)
    elif len(orig_list.shape) == 2:
        orig_list_len = orig_list.shape[0]
        target_list = []
        for i in range(orig_list.shape[1]):
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
            f = interpolate.interp1d(x, orig_list[:, i], 'linear')
            del x
            resample_list = f([x for x in range(target_len)])
            target_list.append(resample_list)
        resample_list = np.asarray(target_list).T
    elif len(orig_list.shape) == 3:
        orig_list_len = orig_list.shape[0]
        target_list = [([]) for _ in range(orig_list.shape[2])]
        for j in range(orig_list.shape[2]):
            for i in range(orig_list.shape[1]):
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
                f = interpolate.interp1d(x, orig_list[:, i, j], 'linear')
                del x
                resample_list = f([x for x in range(target_len)])
                target_list[j].append(resample_list)
        resample_list = np.asarray(target_list).T
    else:
        print(len(orig_list.shape))
        return 0
    return resample_list


def optimization_pyomo_squat(arm, torque, emg, iso, flfv):
    shape0 = arm.shape[0]
    shape1 = arm.shape[1]
    shape2 = arm.shape[2]

    # 创建一个具体的模型
    model = ConcreteModel()
    model.I = RangeSet(0, shape0)
    model.J = RangeSet(0, shape1)
    model.K = RangeSet(0, shape2)

    # 定义变量
    model.x = Var(model.I, model.J, within=NonNegativeReals)  # activation
    model.y = Var(model.J, within=NonNegativeReals)  # maximum isometric force
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t = Var(model.I, model.K, within=Reals)  # torque

    # 定义约束条件
    model.constr = ConstraintList()
    for j in range(shape1):
        model.constr.add(model.y[j] >= iso[j] * 0.05)
        model.constr.add(model.y[j] <= iso[j] * 200)
    for i in range(shape0):
        for j in range(shape1):
            model.constr.add(model.x[i, j] >= emg[i, j] * 0.80)
            model.constr.add(model.x[i, j] <= emg[i, j] * 1.20)
            # model.constr.add(model.x[i, j] >= emg_mean[i - 1, j - 1] - emg_std[i - 1, j - 1] * 2.5)
            # model.constr.add(model.x[i, j] <= emg_mean[i - 1, j - 1] + emg_std[i - 1, j - 1] * 2.5)
            model.constr.add(model.x[i, j] >= 0)
            model.constr.add(model.x[i, j] <= 1)
            model.constr.add(model.f[i, j] == model.x[i, j] * model.y[j])  # muscle force
            # model.constr.add(model.f[i, j] == model.x[i, j] * model.y[j] * flfv[i, j])  # muscle force
        for k in range(shape2):
            model.constr.add(model.t[i, k] == sum(model.f[i, j] * arm[i, j, k] for j in range(shape1)))

    # objective function
    obj = sum((sum((model.t[i, k] - torque[i, k]) ** 2 for i in range(shape0)) / shape0) for k in range(shape2))
    model.obj = Objective(expr=obj, sense=minimize)

    solver = SolverFactory('ipopt')

    results = []

    def solve_optimization():
        result = solver.solve(model)
        results.append(value(model.obj))
        return result

    results = solver.solve(model)

    c = np.ones_like(emg)
    d = np.ones_like(iso)
    e = np.ones_like(torque)
    for i in range(shape0):
        for j in range(shape1):
            c[i, j] = value(model.x[i, j])
    for j in range(shape1):
        d[j] = value(model.y[j])
    for i in range(shape0):
        for k in range(shape2):
            e[i, k] = value(model.t[i, k])
    x = c
    y = d
    t = e

    # 打印结果
    for i in range(total_muscle_num):
        print("{:.2f}".format(np.asarray(y)[i]))  # print the maximum isometric force

    active_force = emg * y
    calu_torque = []
    for k in range(shape2):
        calu_torque.append([sum(active_force[:, j] * arm[:, j, k]) for j in range(shape1)])
        rmse = np.sqrt(np.sum((np.asarray(t[:, k]) - torque[:, k]) ** 2) / len(torque[:, k]))
        print("torque rmse:\t", "{:.2f}".format(rmse))

    # plt.figure()
    # for i in range(4):
    #     plt.subplot(4, 1, i + 1)
    #     plt.plot(t[:, i])
    #     plt.plot(torque[:, i])
    # plt.show()

    output = {
        'torque': t,
        'activation': x,
        'fmax': y,
        'calu_torque': calu_torque
    }
    return output


def optimization_test_squat(arm, torque, emg, y, mean, std):
    shape0 = arm.shape[0]
    shape1 = arm.shape[1]
    shape2 = arm.shape[2]

    # 创建一个具体的模型
    model = ConcreteModel()
    model.I = RangeSet(0, shape0)
    model.J = RangeSet(0, shape1)
    model.K = RangeSet(0, shape2)

    # 定义变量
    model.x = Var(model.I, model.J, within=NonNegativeReals)  # activation
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t = Var(model.I, model.K, within=Reals)  # torque

    # 定义约束条件
    model.constr = ConstraintList()
    for i in range(shape0):
        for j in range(shape1):
            # model.constr.add(model.x[i, j] >= emg[i, j] * 0.80)
            # model.constr.add(model.x[i, j] <= emg[i, j] * 1.20)
            model.constr.add(model.x[i, j] >= mean[i, j] - std[i, j] * 2.5)
            model.constr.add(model.x[i, j] <= mean[i, j] + std[i, j] * 2.5)
            model.constr.add(model.x[i, j] >= 0)
            model.constr.add(model.x[i, j] <= 1)
            model.constr.add(model.f[i, j] == model.x[i, j] * y[j])  # muscle force
            # model.constr.add(model.f[i, j] == model.x[i, j] * model.y[j] * flfv[i, j])  # muscle force
        for k in range(shape2):
            model.constr.add(model.t[i, k] == sum(model.f[i, j] * arm[i, j, k] for j in range(shape1)))

    # objective function
    obj = sum((sum((model.t[i, k] - torque[i, k]) ** 2 for i in range(shape0)) / shape0) for k in range(shape2))
    model.obj = Objective(expr=obj, sense=minimize)

    solver = SolverFactory('ipopt')

    results = []

    def solve_optimization():
        result = solver.solve(model)
        results.append(value(model.obj))
        return result

    results = solver.solve(model)

    c = np.ones_like(emg)
    e = np.ones_like(torque)
    for i in range(shape0):
        for j in range(shape1):
            c[i, j] = value(model.x[i, j])
    for i in range(shape0):
        for k in range(shape2):
            e[i, k] = value(model.t[i, k])
    x = c
    t = e

    # # 打印结果
    # for i in range(total_muscle_num):
    #     print("{:.2f}".format(np.asarray(y)[i]))  # print the maximum isometric force

    active_force = emg * y
    calu_torque = []
    print('-'*25, 'test', '-'*25)
    for k in range(shape2):
        calu_torque.append([sum(active_force[:, j] * arm[:, j, k]) for j in range(shape1)])
        rmse = np.sqrt(np.sum((np.asarray(t[:, k]) - torque[:, k]) ** 2) / len(torque[:, k]))
        print("torque rmse:\t", "{:.2f}".format(rmse))

    # plt.figure()
    # for i in range(4):
    #     plt.subplot(4, 1, i + 1)
    #     plt.plot(t[:, i])
    #     plt.plot(torque[:, i])
    # plt.show()

    output = {
        'torque': t,
        'activation': x,
        'fmax': y,
        'calu_torque': calu_torque
    }
    return output


def plot_all_result_squat(num, t, a, torque, emg, time, mean, std, idx=None):
    for i in range(num):
        plot_result_squat(i, t, a, torque, emg, time, mean, std, idx)


def plot_result_squat(num, t, a, torque, emg, time, mean, std, idx=None):
    len1 = length
    t = t[num * len1:(num + 1) * len1, :]
    a = a[num * len1:(num + 1) * len1, :]
    emg = emg[num * len1:(num + 1) * len1, :]
    time = time[num * len1:(num + 1) * len1, :]
    torque = torque[num * len1:(num + 1) * len1, :]
    # calu_torque1 = calu_torque1[num * len1:(num + 1) * len1]
    # emg_std_long = emg_std_long[:, num * len2:(num + 1) * len2]
    # emg_mean_long = emg_mean_long[:, num * len2:(num + 1) * len2]
    # time_long = resample_by_len(list(time), emg_mean_long.shape[1])
    if idx is None:
        idx = num

    # plt.figure(figsize=(3.3, 7.7))
    # plt.subplots_adjust(left=0.225, right=0.935)

    # plt.figure(figsize=(6, 6))
    # for i in range(len(TorqueList)):
    #     plt.subplot(len(TorqueList), 1, i + 1)
    #     plt.plot(time, torque[:, i], label='measured', linewidth=2)
    #     plt.plot(time, np.asarray(t[:, i]), label='optimization', linewidth=2)
    #     plt.legend()
    #     plt.ylabel('torque' + str(i), weight='bold', size=10)
    # plt.xlabel('time (s)', weight='bold', size=10)

    # plt.figure(figsize=(6, 6))
    if num == 0:
        plt.figure(figsize=(2.5, 6.7))
        plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    else:
        plt.figure(figsize=(2.3, 6.7))
        plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)
    # plt.figure(figsize=(2.5, 6.7))
    for i in range(len(TorqueList) + 6):
        if i < len(TorqueList):
            plt.subplot(len(TorqueList) + 6, 1, i + 1)
            plt.plot(time, np.asarray(t[:, i]), label='optimization', linewidth=2)
            plt.plot(time, torque[:, i], label='measured', linewidth=2)
            # plt.legend()
            if num == 0:
                plt.ylabel('torque' + str(i + 1), weight='bold', size=10)
        else:
            plt.subplot(len(TorqueList) + 6, 1, i + 1)
            plt.plot(time, np.asarray(a[:, muscle_idx[i - len(TorqueList)]]), label='optimization', linewidth=2,
                     zorder=2)
            if plot_distribution is True:
                plt.errorbar(time, mean[:, muscle_idx[i - len(TorqueList)]], 2 * std[:, muscle_idx[i - len(TorqueList)]],
                             label='emg', color='lavender', zorder=1)
            else:
                plt.plot(time, np.asarray(emg[:, muscle_idx[i - len(TorqueList)]]), label='emg', linewidth=2, zorder=3)
            # if legend_label is True or num == 0:
            #     plt.ylabel(musc_label[j], weight='bold')
            if num == 0:
                plt.ylabel(musc_label[i - len(TorqueList)], weight='bold', size=10)
            if i - len(TorqueList) == 5:
                plt.xlabel('time (s)', weight='bold', size=10)
            else:
                ax = plt.gca()
                ax.set_xticklabels([])
            # plt.legend()
    plt.savefig('squat_train_{}.png'.format(idx))

    # if num == 0:
    #     plt.figure(figsize=(2.5, 6.7))
    #     plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    # else:
    #     plt.figure(figsize=(2.3, 6.7))
    #     plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)

    # plt.figure(figsize=(6, 8))
    # for j in range(6):
    #     plt.subplot(6, 1, j + 1)
    #     plt.plot(time, np.asarray(emg[:, muscle_idx[j]]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(a[:, muscle_idx[j]]), label='optimization', linewidth=2, zorder=2)
    #     # if legend_label is True or num == 0:
    #     #     plt.ylabel(musc_label[j], weight='bold')
    #     if j == 5:
    #         plt.xlabel('time (s)')
    #     else:
    #         ax = plt.gca()
    #         ax.set_xticklabels([])
    #     plt.legend()
    # plt.savefig('squat_train_{}.png'.format(idx))


def plot_test_result_squat(num, t, a, torque, emg, time, mean, std):
    idx = num

    # plt.figure(figsize=(6, 6))
    # plt.figure(figsize=(2.3, 6.7))
    # plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)
    plt.figure(figsize=(2.5, 6.7))
    plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    for i in range(len(TorqueList) + 6):
        if i < len(TorqueList):
            plt.subplot(len(TorqueList) + 6, 1, i + 1)
            plt.plot(time, np.asarray(t[:, i]), label='optimization', linewidth=2)
            plt.plot(time, torque[:, i], label='measured', linewidth=2)
            # plt.legend()
            plt.ylabel('torque' + str(i + 1), weight='bold', size=10)
        else:
            plt.subplot(len(TorqueList) + 6, 1, i + 1)
            plt.errorbar(time, mean[:, muscle_idx[i - len(TorqueList)]], 2 * std[:, muscle_idx[i - len(TorqueList)]],
                         label='emg', color='lavender', zorder=1)
            plt.plot(time, np.asarray(a[:, muscle_idx[i - len(TorqueList)]]), label='optimization', linewidth=2,
                     zorder=2)
            plt.plot(time, np.asarray(emg[:, muscle_idx[i - len(TorqueList)]]), label='emg', linewidth=2, zorder=3)
            plt.ylabel(musc_label[i - len(TorqueList)], weight='bold', size=10)
            # if legend_label is True or num == 0:
            #     plt.ylabel(musc_label[j], weight='bold')
            if i - len(TorqueList) == 5:
                plt.xlabel('time (s)', weight='bold', size=10)
            else:
                ax = plt.gca()
                ax.set_xticklabels([])
            # plt.legend()
    plt.savefig('squat_test_{}.png'.format(idx))

    # if num == 0:
    #     plt.figure(figsize=(2.5, 6.7))
    #     plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    # else:
    #     plt.figure(figsize=(2.3, 6.7))
    #     plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)

    # plt.figure(figsize=(6, 8))
    # for j in range(6):
    #     plt.subplot(6, 1, j + 1)
    #     plt.plot(time, np.asarray(emg[:, muscle_idx[j]]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(a[:, muscle_idx[j]]), label='optimization', linewidth=2, zorder=2)
    #     # if legend_label is True or num == 0:
    #     #     plt.ylabel(musc_label[j], weight='bold')
    #     if j == 5:
    #         plt.xlabel('time (s)')
    #     else:
    #         ax = plt.gca()
    #         ax.set_xticklabels([])
    #     plt.legend()
    # plt.savefig('squat_train_{}.png'.format(idx))


def plot_before_result_squat(num, t, a, torque, emg, time):
    idx = num

    # plt.figure(figsize=(6, 6))
    plt.figure(figsize=(2.5, 6.7))
    plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    for i in range(len(TorqueList) + 6):
        if i < len(TorqueList):
            plt.subplot(len(TorqueList) + 6, 1, i + 1)
            plt.plot(time, np.asarray(t[:, i]), label='optimization', linewidth=2)
            plt.plot(time, torque[:, i], label='measured', linewidth=2)

            # plt.legend()
            plt.ylabel('torque' + str(i), weight='bold', size=10)
        else:
            plt.subplot(len(TorqueList) + 6, 1, i + 1)
            plt.plot(time, np.asarray(a[:, muscle_idx[i - len(TorqueList)]]), label='optimization', linewidth=2,
                     zorder=2)
            plt.plot(time, np.asarray(emg[:, muscle_idx[i - len(TorqueList)]]), label='emg', linewidth=2, zorder=3)
            plt.ylabel(musc_label[i - len(TorqueList)], weight='bold', size=10)
            if i - len(TorqueList) == 5:
                plt.xlabel('time (s)', weight='bold', size=10)
            else:
                ax = plt.gca()
                ax.set_xticklabels([])
            # plt.legend()
    plt.savefig('squat_before_{}.png'.format(idx))

    # if num == 0:
    #     plt.figure(figsize=(2.5, 6.7))
    #     plt.subplots_adjust(left=0.285, right=0.985, top=0.990, bottom=0.075)
    # else:
    #     plt.figure(figsize=(2.3, 6.7))
    #     plt.subplots_adjust(left=0.225, right=0.985, top=0.990, bottom=0.075)

    # plt.figure(figsize=(6, 8))
    # for j in range(6):
    #     plt.subplot(6, 1, j + 1)
    #     plt.plot(time, np.asarray(emg[:, muscle_idx[j]]), label='emg', linewidth=2, zorder=3)
    #     plt.plot(time, np.asarray(a[:, muscle_idx[j]]), label='optimization', linewidth=2, zorder=2)
    #     # if legend_label is True or num == 0:
    #     #     plt.ylabel(musc_label[j], weight='bold')
    #     if j == 5:
    #         plt.xlabel('time (s)')
    #     else:
    #         ax = plt.gca()
    #         ax.set_xticklabels([])
    #     plt.legend()
    # plt.savefig('squat_train_{}.png'.format(idx))


def moving_average_filter(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data


def read_mat(file):
    mat = scipy.io.loadmat(file)
    mat = mat['Trial']
    muscleNames = mat['muscleNames'][0][0][0, :]
    torqueName = mat['torqueName'][0][0]
    coordinateNames = mat['coordinateNames'][0][0].T
    time = mat['time'][0][0].T
    torque = mat['torque'][0][0]
    flMultiplier = mat['flMultiplier'][0][0]
    fvMultiplier = mat['fvMultiplier'][0][0]
    PassiveflMultiplier = mat['PassiveflMultiplier'][0][0]
    cosAlphaMultiplier = mat['cosAlphaMultiplier'][0][0]
    MmtArm = mat['MmtArm'][0][0]
    activation = mat['Activation'][0][0]
    ApplyForce = mat['ApplyForce'][0][0]

    # TorqueList = [6, 7, 8, 9, 10, 11, 12, 13]
    # TarMuscleList = [EMG1Group, EMG2Group, EMG3Group, EMG4Group, EMG5Group, EMG6Group,
    #                  EMG7Group, EMG8Group, EMG9Group, EMG10Group, EMG11Group, EMG12Group]

    # musclelist = []
    # for i in TarMuscleList:
    #     GroupList = []
    #     for j in i:
    #         indices = np.where(muscleNames == j)[0]
    #         GroupList.append(indices)
    #     musclelist.append(np.asarray(GroupList))

    fmaxinit = scipy.io.loadmat(folder + 'fmaxinit.mat')['F_Max_init'].squeeze()
    musclelist = []
    iso = []
    for i in TarMuscleList:
        for j in i:
            indices = np.where(muscleNames == j)[0]
            musclelist.append(indices)
            iso.append(fmaxinit[indices])
    musclelist = np.asarray(musclelist).squeeze()
    iso = np.asarray(iso).squeeze()

    # torque = torque[:, TorqueList].T
    # activation = activation[:, musclelist].T
    # MmtArm = MmtArm[:, musclelist][:, :, TorqueList].T
    # flMultiplier = flMultiplier[:, musclelist].T
    # fvMultiplier = fvMultiplier[:, musclelist].T
    # PassiveflMultiplier = PassiveflMultiplier[:, musclelist].T
    # cosAlphaMultiplier = cosAlphaMultiplier[:, musclelist].T

    torque = torque[:, TorqueList]
    activation = activation[:, musclelist]
    MmtArm = MmtArm[:, musclelist][:, :, TorqueList]
    flMultiplier = flMultiplier[:, musclelist]
    fvMultiplier = fvMultiplier[:, musclelist]
    PassiveflMultiplier = PassiveflMultiplier[:, musclelist]
    cosAlphaMultiplier = cosAlphaMultiplier[:, musclelist]

    window_size = 3
    for i in range(torque.shape[1]):
        # [b, a] = scipy.signal.butter(4, 0.2, 'low')
        # [b, a] = scipy.signal.butter(4, 0.2, 'high')
        # torque[:, i] = signal.filtfilt(b, a, torque[:, i])
        torque[:, i] = moving_average_filter(torque[:, i], window_size)

    output = {
        # 'muscleList': musclelist,
        # 'torqueList': TorqueList,
        'time': resample_by_len(time, length),
        'torque': resample_by_len(torque, length),
        'fl': resample_by_len(flMultiplier, length),
        'fv': resample_by_len(fvMultiplier, length),
        'flfv': resample_by_len(flMultiplier * fvMultiplier, length),
        'Pfl': resample_by_len(PassiveflMultiplier, length),
        'cosAlpha': resample_by_len(cosAlphaMultiplier, length),
        'MmtArm': resample_by_len(MmtArm, length),
        'activation': resample_by_len(activation, length),
        'iso': iso
        # 'ApplyForce': ApplyForce
    }
    return output


def read_all_mat(label='training'):
    if label == 'training':
        mat1 = read_mat(folder + '20kg_div1.mat')
        mat2 = read_mat(folder + '20kg_div2.mat')
        mat3 = read_mat(folder + '20kg_div3.mat')
        mat4 = read_mat(folder + '30kg_div1.mat')
        mat5 = read_mat(folder + '30kg_div2.mat')
        mat6 = read_mat(folder + '30kg_div3.mat')
        mat7 = read_mat(folder + '35kg_div1.mat')
        mat8 = read_mat(folder + '35kg_div2.mat')
        mat9 = read_mat(folder + '35kg_div3.mat')
        mats = [mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9]
    elif label == 'all':
        mat1 = read_mat(folder + '20kg_div1.mat')
        mat2 = read_mat(folder + '20kg_div2.mat')
        mat3 = read_mat(folder + '20kg_div3.mat')
        mat4 = read_mat(folder + '20kg_div4.mat')
        mat5 = read_mat(folder + '20kg_div5.mat')
        mat6 = read_mat(folder + '30kg_div1.mat')
        mat7 = read_mat(folder + '30kg_div2.mat')
        mat8 = read_mat(folder + '30kg_div3.mat')
        mat9 = read_mat(folder + '30kg_div4.mat')
        mat10 = read_mat(folder + '30kg_div5.mat')
        mat11 = read_mat(folder + '35kg_div1.mat')
        mat12 = read_mat(folder + '35kg_div2.mat')
        mat13 = read_mat(folder + '35kg_div3.mat')
        mat14 = read_mat(folder + '35kg_div4.mat')
        mat15 = read_mat(folder + '35kg_div5.mat')
        mats = [mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11, mat12, mat13, mat14, mat15]
    # mats = [mat1, mat2, mat3]
    output = {}
    for j in mat1:
        if j == 'iso':
            output.update({j: mat1[j]})
        else:
            output.update({j: np.concatenate([mats[i][j] for i in range(len(mats))], axis=0)})
    return output


def emg_distribution():
    mat = read_all_mat('all')
    emg = mat['activation']
    num = int(emg.shape[0] / length)

    def cidx(n):
        idx = []
        n = n % length
        for j in range(num):
            idx.append(j * length + n)
        return np.asarray(idx)

    data_mean = [([]) for _ in range(emg.shape[1])]
    data_std = [([]) for _ in range(emg.shape[1])]

    for j in range(emg.shape[1]):
        for i in range(length):
            id = cidx(i)
            data_mean[j].append(np.mean(emg[id, j]))
            data_std[j].append(np.std(emg[id, j]))

    data_mean = np.asarray(data_mean).T
    data_std = np.asarray(data_std).T
    mat.update({'mean': data_mean})
    mat.update({'std': data_std})
    return mat


def application(y, mean, std):
    # files = ['20kg_div4', '20kg_div5', '30kg_div4', '30kg_div5', '35kg_div4', '35kg_div5']
    files = ['35kg_div1', '35kg_div2', '35kg_div3', '35kg_div4', '35kg_div5']
    for f in range(len(files)):
        mat = read_mat(folder + files[f] + '.mat')
        opt = optimization_test_squat(mat['MmtArm'], mat['torque'], mat['activation'], y, mean, std)
        plot_test_result_squat(f, opt['torque'], opt['activation'], mat['torque'], mat['activation'], mat['time'],
                               mean, std)


def before_motion(y, mean, std):
    files1 = ['30kg_div1', '30kg_div2', '30kg_div3', '30kg_div4', '30kg_div5']
    files = ['35kg_div1', '35kg_div2', '35kg_div3', '35kg_div4', '35kg_div5']
    for f in range(len(files)):
        mat1 = read_mat(folder + files1[f] + '.mat')
        mat = read_mat(folder + files[f] + '.mat')
        opt = optimization_test_squat(mat1['MmtArm'], mat1['torque'], mat1['activation'], y, mean, std)
        plot_before_result_squat(f, opt['torque'], opt['activation'], mat1['torque'], mat['activation'], mat['time'])


if __name__ == '__main__':
    mat = read_all_mat('training')
    dis = emg_distribution()
    opt = optimization_pyomo_squat(mat['MmtArm'], mat['torque'], mat['activation'], mat['iso'], mat['fv'])
    plot_all_result_squat(int(mat['torque'].shape[0] / length), opt['torque'], opt['activation'], mat['torque'],
                          mat['activation'], mat['time'], dis['mean'], dis['std'])
    before_motion(opt['fmax'], dis['mean'], dis['std'])
    application(opt['fmax'], dis['mean'], dis['std'])

    plt.show()
