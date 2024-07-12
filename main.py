import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from pyomo.environ import *
from pyomo.opt import SolverFactory

from emg_distribution import *
from test import *
from require import *


def optimization(emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u,
                 emg_trend_d):
    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    y = m.Array(m.Var, len(muscle_idx), lb=0)  # maximum isometric force
    c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    f = m.Array(m.Var, arm.shape)  # muscle force
    t = m.Array(m.Var, torque.shape)  # torque
    # k = m.Array(m.Var, 4)  # fv的阶数（n-1）
    m.Minimize(np.square(t - torque).mean())
    for i in range(arm.shape[0]):
        for j in range(arm.shape[1]):
            expr1 = x[i][j] * y[i] == f[i, j]
            # expr1 = x[i][j] * y[i] * fa[i][j] == f[i, j]
            m.Equation(expr1)
            if i == arm.shape[0] - 1:
                expr2 = sum(f[:, j] * arm[:, j]) == t[j]
                m.Equation(expr2)
            if mvc_is_variable is True:
                m.Equation(x[i][j] <= (emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2))
                m.Equation(x[i][j] >= (emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2))
                m.Equation((emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2) <= 1)
                # m.Equation((emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2) >= 0)
            else:
                if emg[i][j] < 0.01:
                    m.Equation(x[i][j] <= (emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2))
                    m.Equation(x[i][j] >= (emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2))
                else:
                    m.Equation(x[i][j] <= emg[i][j] * 1.20)
                    m.Equation(x[i][j] >= emg[i][j] * 0.80)
                # m.Equation(x[i][j] <= (emg_mean[i][j] + emg_std[i][j] * 2))
                # m.Equation(x[i][j] >= (emg_mean[i][j] - emg_std[i][j] * 2))
        # m.Equations([y[i] >= 3 * iso1[i], y[i] <= 3 * iso2[i]])
        # m.Equations([y[i] >= iso1[i], y[i] <= iso2[i]])
        m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 200 * iso[i]])
        # m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 3 * iso[i]])
        if mvc_is_variable is True:
            m.Equations([c[i] >= 0.01, c[i] <= 5])
    # m.Equations([y[0] >= 0.5 * y[1], y[0] <= 2.0 * y[1]])
    # m.Equations([y[2] >= 0.1 * y[0], y[2] >= 0.1 * y[1]])
    # m.options.MAX_ITER = 10000
    m.solve(disp=False)
    # print(x)
    # print(y)
    for i in range(y.size):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i].value[0]))
    if mvc_is_variable is True:
        print('-' * 50)
        for i in range(c.size):
            print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(c)[i].value[0]))

    r = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i][j] = r[i][j] * x[i][j].value[0]
    e = np.ones_like(t)
    for i in range(t.shape[0]):
        e[i] = e[i] * t[i].value[0]
    t = e

    c_r = [(np.asarray(c)[i].value[0]) for i in range(c.size)]
    y_r = [(np.asarray(y)[i].value[0]) for i in range(y.size)]  # maximum isometric force
    if mvc_is_variable is True:
        active_force = emg.T * c_r * y_r
    else:
        active_force = emg.T * y_r
        # active_force = fa.T * emg.T * y_r
    calu_torque = [sum(active_force[j, :] * arm[:, j]) for j in range(arm.shape[1])]
    rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    return t, r, calu_torque, c, y_r


def optimization_bp(emg_mean, emg_std, arm1, arm2, torque1, torque2, time, emg):
    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm1.shape, lb=0, ub=1)  # activation
    y = m.Array(m.Var, len(muscle_idx), lb=0)  # maximum isometric force
    c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    f = m.Array(m.Var, arm1.shape)  # muscle force
    t1 = m.Array(m.Var, torque1.shape)  # torque
    t2 = m.Array(m.Var, torque2.shape)  # torque
    # k = m.Array(m.Var, 4)  # fv的阶数（n-1）
    m.Minimize(np.square(t1 - torque1).mean() + np.square(t2 - torque2).mean())
    for i in range(arm1.shape[0]):
        for j in range(arm1.shape[1]):
            expr1 = x[i][j] * y[i] == f[i, j]
            # expr1 = x[i][j] * y[i] * fa[i][j] == f[i, j]
            m.Equation(expr1)
            if i == arm1.shape[0] - 1:
                expr2 = sum(f[:, j] * arm1[:, j]) == t1[j]
                expr3 = sum(f[:, j] * arm2[:, j]) == t2[j]
                m.Equation(expr2)
                m.Equation(expr3)
            if mvc_is_variable is True:
                m.Equation(x[i][j] <= (emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2))
                m.Equation(x[i][j] >= (emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2))
                m.Equation((emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2) <= 1)
                # m.Equation((emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2) >= 0)
            else:
                m.Equation(x[i][j] <= emg[i][j] * 1.20)
                m.Equation(x[i][j] >= emg[i][j] * 0.80)
                # m.Equation(x[i][j] <= (emg_mean[i][j] + emg_std[i][j] * 2))
                # m.Equation(x[i][j] >= (emg_mean[i][j] - emg_std[i][j] * 2))
        # m.Equations([y[i] >= 3 * iso1[i], y[i] <= 3 * iso2[i]])
        # m.Equations([y[i] >= iso1[i], y[i] <= iso2[i]])
        m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 100 * iso[i]])
        # m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 3 * iso[i]])
        if mvc_is_variable is True:
            m.Equations([c[i] >= 0.01, c[i] <= 5])
    # m.Equations([y[0] >= 0.5 * y[1], y[0] <= 2.0 * y[1]])
    # m.Equations([y[2] >= 0.1 * y[0], y[2] >= 0.1 * y[1]])
    # m.options.MAX_ITER = 10000
    m.solve(disp=False)
    # print(x)
    # print(y)
    for i in range(y.size):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i].value[0]))
    if mvc_is_variable is True:
        print('-' * 50)
        for i in range(c.size):
            print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(c)[i].value[0]))

    r = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i][j] = r[i][j] * x[i][j].value[0]
    e1 = np.ones_like(t1)
    e2 = np.ones_like(t2)
    for i in range(t1.shape[0]):
        e1[i] = e1[i] * t1[i].value[0]
        e2[i] = e2[i] * t2[i].value[0]
    t1 = e1
    t2 = e2

    c_r = [(np.asarray(c)[i].value[0]) for i in range(c.size)]
    y_r = [(np.asarray(y)[i].value[0]) for i in range(y.size)]  # maximum isometric force
    if mvc_is_variable is True:
        active_force = emg.T * c_r * y_r
    else:
        active_force = emg.T * y_r
        # active_force = fa.T * emg.T * y_r
    calu_torque1 = [sum(active_force[j, :] * arm1[:, j]) for j in range(arm1.shape[1])]
    calu_torque2 = [sum(active_force[j, :] * arm2[:, j]) for j in range(arm2.shape[1])]
    rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    return t1, t2, r, calu_torque1, calu_torque2, c, y_r


def optimization_pyomo(emg_mean, emg_std, arm, torque, time, emg):
    shape0 = arm.shape[0]
    shape1 = arm.shape[1]

    # 创建一个具体的模型
    model = ConcreteModel()
    model.I = RangeSet(0, shape0 - 1)
    model.J = RangeSet(0, shape1 - 1)

    # 定义变量
    model.x = Var(model.I, model.J, within=NonNegativeReals)  # activation
    model.y = Var(model.I, within=NonNegativeReals)  # maximum isometric force
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t = Var(model.J, within=Reals)  # torque

    try:
        # 定义约束条件
        model.constr = ConstraintList()
        # iso_opti = iso[:shape0]
        for i in model.I:
            model.constr.add(model.y[i] >= iso_min[i])
            model.constr.add(model.y[i] <= iso_max[i])
        for j in model.J:
            for i in model.I:
                # model.constr.add(model.x[i, j] >= emg[i, j] * 0.80)
                # model.constr.add(model.x[i, j] <= emg[i, j] * 1.20)
                model.constr.add(model.x[i, j] >= emg_mean[i, j] - emg_std[i, j] * 2.5)
                model.constr.add(model.x[i, j] <= emg_mean[i, j] + emg_std[i, j] * 2.5)
                model.constr.add(model.x[i, j] >= emg[i, j] - emg_std[i, j] * 2.5)
                model.constr.add(model.x[i, j] <= emg[i, j] + emg_std[i, j] * 2.5)
                # con = (emg_mean[i, j] + emg_std[i, j] * 2.5 <= emg[i, j] - emg_std[i, j] * 2.5
                #        or emg_mean[i, j] - emg_std[i, j] * 2.5 >= emg[i, j] + emg_std[i, j] * 2.5)
                # if con is False:
                #     model.constr.add(model.x[i, j] >= emg_mean[i, j] - emg_std[i, j] * 2.5)
                #     model.constr.add(model.x[i, j] <= emg_mean[i, j] + emg_std[i, j] * 2.5)
                #     model.constr.add(model.x[i, j] >= emg[i, j] - emg_std[i, j] * 2.5)
                #     model.constr.add(model.x[i, j] <= emg[i, j] + emg_std[i, j] * 2.5)
                # else:
                #     model.constr.add(model.x[i, j] >= emg[i, j] - emg_std[i, j] * 2)
                #     model.constr.add(model.x[i, j] <= emg[i, j] + emg_std[i, j] * 2)
                model.constr.add(model.x[i, j] >= 0)
                model.constr.add(model.x[i, j] <= 1)
                model.constr.add(model.f[i, j] == model.x[i, j] * model.y[i])  # muscle force
            model.constr.add(model.t[j] == sum(model.f[i, j] * arm[i, j] for i in model.I))  # torque

        # 定义目标函数
        # obj = sum(np.square(model.t[j] - torque[j - 1]) for j in model.J)
        obj = sum((model.t[j] - torque[j]) ** 2 for j in model.J)
        model.obj = Objective(expr=obj, sense=minimize)

        # 求解器配置
        solver = SolverFactory('ipopt')
    except Warning:
        print('warning')

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

    # 打印优化结果
    # model.pprint()
    # model.x.pprint()
    # model.y.pprint()

    c = np.ones_like(arm)
    d = np.ones_like(iso_max)
    e = np.ones_like(torque)
    for i in model.I:
        for j in model.J:
            c[i, j] = value(model.x[i, j])
    for i in model.I:
        d[i] = value(model.y[i])
    for j in model.J:
        e[j] = value(model.t[j])
    x = c
    y = d
    t = e

    # # 打印结果
    # print("最优解：x =", x, "y =", y)

    # 打印结果
    for i in range(len(muscle_idx)):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i]))  # print the maximum isometric force

    active_force = emg.T * y
    calu_torque = [sum(active_force[j, :] * arm[:, j]) for j in range(arm.shape[1])]
    calu_torque = np.asarray(calu_torque)

    rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
    print("torque rmse:\t", "{:.2f}".format(rmse))
    # plt.figure()
    # plt.plot(iter_obj)
    # plt.show()
    return t, x, calu_torque, y


def optimization_pyomo_bp(emg_mean, emg_std, arm1, arm2, torque1, torque2, time, emg):
    shape0 = arm1.shape[0]
    shape1 = arm1.shape[1]

    # 创建一个具体的模型
    model = ConcreteModel()
    model.I = RangeSet(1, shape0)
    model.J = RangeSet(1, shape1)

    # 定义变量
    model.x = Var(model.I, model.J, within=NonNegativeReals)  # activation
    model.y = Var(model.I, within=NonNegativeReals)  # maximum isometric force
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t1 = Var(model.J, within=Reals)  # torque
    model.t2 = Var(model.J, within=Reals)  # torque

    # 定义约束条件
    model.constr = ConstraintList()
    for i in model.I:
        model.constr.add(model.y[i] >= iso[i - 1] * 0.05)
        model.constr.add(model.y[i] <= iso[i - 1] * 200)
    for j in model.J:
        for i in model.I:
            model.constr.add(model.x[i, j] >= emg[i - 1, j - 1] * 0.80)
            model.constr.add(model.x[i, j] <= emg[i - 1, j - 1] * 1.20)
            # model.constr.add(model.x[i, j] >= emg_mean[i - 1, j - 1] - emg_std[i - 1, j - 1] * 2.5)
            # model.constr.add(model.x[i, j] <= emg_mean[i - 1, j - 1] + emg_std[i - 1, j - 1] * 2.5)
            model.constr.add(model.x[i, j] >= 0)
            model.constr.add(model.x[i, j] <= 1)
            model.constr.add(model.f[i, j] == model.x[i, j] * model.y[i])  # muscle force
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

    # 打印优化结果
    # model.pprint()
    # model.x.pprint()
    # model.y.pprint()

    c = np.ones_like(arm1)
    d = np.ones_like(iso)
    e1 = np.ones_like(torque1)
    e2 = np.ones_like(torque2)
    for i in model.I:
        for j in model.J:
            c[i - 1, j - 1] = value(model.x[i, j])
    for i in model.I:
        d[i - 1] = value(model.y[i])
    for j in model.J:
        e1[j - 1] = value(model.t1[j])
        e2[j - 1] = value(model.t2[j])
    x = c
    y = d
    t1 = e1
    t2 = e2

    # # 打印结果
    # print("最优解：x =", x, "y =", y)

    # 打印结果
    for i in range(len(measured_muscle_idx)):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i]))  # print the maximum isometric force

    active_force = emg.T * y
    calu_torque1 = [sum(active_force[j, :] * arm1[:, j]) for j in range(arm1.shape[1])]
    calu_torque2 = [sum(active_force[j, :] * arm2[:, j]) for j in range(arm2.shape[1])]

    rmse1 = np.sqrt(np.sum((np.asarray(t1) - torque1) ** 2) / len(torque1))
    rmse2 = np.sqrt(np.sum((np.asarray(t2) - torque2) ** 2) / len(torque2))
    # print("torque rmse:\t", "{:.2f}".format(rmse1))
    # print("torque rmse:\t", "{:.2f}".format(rmse2))
    # plt.figure()
    # plt.plot(iter_obj)
    # plt.show()
    return t1, t2, x, calu_torque1, calu_torque2, y


def optimization_pyomo_deadlift(emg_mean, emg_std, arm, torque, emg):
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
    model.y = Var(model.I, within=NonNegativeReals)  # maximum isometric force
    model.f = Var(model.I, model.J, within=Reals)  # muscle force
    model.t = Var(model.J, model.K, within=Reals)  # torque

    # 定义约束条件
    model.constr = ConstraintList()
    for i in model.I:
        model.constr.add(model.y[i] >= iso_min[i])
        model.constr.add(model.y[i] <= iso_max[i])
    for j in model.J:
        for i in model.I:
            model.constr.add(model.x[i, j] >= emg_mean[i, j] - emg_std[i, j] * 2.5)
            model.constr.add(model.x[i, j] <= emg_mean[i, j] + emg_std[i, j] * 2.5)
            model.constr.add(model.x[i, j] >= emg[i, j] - emg_std[i, j] * 2.5)
            model.constr.add(model.x[i, j] <= emg[i, j] + emg_std[i, j] * 2.5)
            model.constr.add(model.x[i, j] >= 0)
            model.constr.add(model.x[i, j] <= 1)
            model.constr.add(model.f[i, j] == model.x[i, j] * model.y[i])  # muscle force
        for k in range(shape2):
            model.constr.add(model.t[j, k] == sum(arm[i, j, k] * model.f[i, j] for i in model.I))  # torque

    # objective function
    obj = sum((sum((model.t[j, k] - torque[j, k]) ** 2 for j in model.J) / shape1) for k in model.K)
    model.obj = Objective(expr=obj, sense=minimize)

    solver = SolverFactory('ipopt')
    solver.solve(model)

    c = np.ones_like(emg)
    d = np.ones_like(iso)
    e = np.ones_like(torque)
    for i in model.I:
        for j in model.J:
            c[i, j] = value(model.x[i, j])
    for i in model.I:
        d[i] = value(model.y[i])
    for j in model.J:
        for k in model.K:
            e[j, k] = value(model.t[j, k])
    x = c
    y = d
    t = np.asarray(e)

    # 打印结果
    for i in range(len(measured_muscle_idx)):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i]))  # print the maximum isometric force

    active_force = emg.T * y
    calu_torque = []
    for k in range(shape2):
        calu_torque.append([sum(active_force[j, :] * arm[:, j, k]) for j in range(shape1)])
        rmse = np.sqrt(np.sum((t[:, k] - torque[:, k]) ** 2) / len(torque[:, k]))
        print("torque rmse:\t", "{:.2f}".format(rmse))
    rmse = np.sqrt(np.sum((t - torque) ** 2) / torque.size)
    print("torque rmse total:\t", "{:.2f}".format(rmse))
    calu_torque = np.asarray(calu_torque)

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


def optimization_mat(time, torque, fvfl, emg, arm):
    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    y = m.Array(m.Var, muscle_number, lb=0)  # maximum isometric force
    c = m.Array(m.Var, muscle_number)  # MVC emg
    f = m.Array(m.Var, arm.shape)  # muscle force
    t = m.Array(m.Var, torque.shape)  # torque
    # m.Minimize(np.square(x - emg).mean())
    m.Minimize(np.square(t - torque).mean())
    for i in range(arm.shape[0]):
        for j in range(arm.shape[1]):
            m.Equation(x[i][j] * y[i] == f[i, j])
            # m.Equation(x[i][j] * y[i] * fa[i][j] == f[i, j])
            # m.Equation(sum(f[:, j] * arm[:, j] * fvfl[:, j]) == t[j])
            m.Equation(sum(f[:, j] * arm[:, j]) == t[j])
            m.Equation(x[i][j] <= emg[i][j] * 1.20)
            m.Equation(x[i][j] >= emg[i][j] * 0.80)
        # m.Equations([y[i] >= 3 * iso1[i], y[i] <= 3 * iso2[i]])
        # m.Equations([y[i] >= iso1[i], y[i] <= iso2[i]])
        # m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 100 * iso[i]])
        m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 5 * iso[i]])
    # m.Equations([y[0] >= 0.5 * y[1], y[0] <= 2.0 * y[1]])
    # m.Equations([y[2] >= 0.1 * y[0], y[2] >= 0.1 * y[1]])
    # m.options.MAX_ITER = 10000
    m.solve(disp=False)
    # print(x)
    # print(y)
    # for i in range(y.size):
    #     print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i].value[0]))
    for i in range(len(measured_muscle_idx)):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i].value[0]))
    if mvc_is_variable is True:
        print('-' * 50)
        for i in range(c.size):
            print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(c)[i].value[0]))

    r = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i][j] = r[i][j] * x[i][j].value[0]
    e = np.ones_like(t)
    for i in range(t.shape[0]):
        e[i] = e[i] * t[i].value[0]
    t = e

    c_r = [(np.asarray(c)[i].value[0]) for i in range(c.size)]
    y_r = [(np.asarray(y)[i].value[0]) for i in range(y.size)]  # maximum isometric force
    if mvc_is_variable is True:
        active_force = emg.T * c_r * y_r
    else:
        active_force = emg.T * y_r
        # active_force = fa.T * emg.T * y_r
    calu_torque = [sum(active_force[j, :] * arm[:, j]) for j in range(arm.shape[1])]
    return t, r, calu_torque, c, y_r


def one_repetition(label, idx):
    if sport_label == 'bench_press':
        if joint_idx == 'all':
            for i in range(len(idx)):
                print('-' * 25, label, '-', idx[i], '-' * 25)
                o = read_realted_files_bp(label, idx[i])
                opt = optimization_pyomo_deadlift(o['emg_mean'], o['emg_std'], o['arm'], o['torque'], o['emg'])
                plot_all_result(1, opt['torque'], opt['activation'], o['emg'].T, o['time'], o['torque'],
                                o['emg_std_long'], o['emg_mean_long'], opt['calu_torque'])
        else:
            for i in range(len(idx)):
                print('-' * 25, label, '-', idx[i], '-' * 25)
                o = read_realted_files_bp(label, idx[i])
                t1, r, calu_torque1, y_r = \
                    optimization_pyomo(o['emg_mean'], o['emg_std'], o['arm' + joint_num[joint_idx]],
                                       o['torque' + joint_num[joint_idx]], o['time'], o['emg'])
                plot_all_result(1, t1, r, o['emg'], o['time'], o['torque' + joint_num[joint_idx]],
                                o['emg_std_long'], o['emg_mean_long'], calu_torque1)
    if sport_label == 'deadlift':
        if joint_idx == 'all':
            for i in range(len(idx)):
                print('-' * 25, label, '-', idx[i], '-' * 25)
                o = read_realted_files_dl(label, idx[i])
                opt = optimization_pyomo_deadlift(o['emg_mean'], o['emg_std'], o['arm'], o['torque'], o['emg'])
                plot_all_result(1, opt['torque'], opt['activation'], o['emg'].T, o['time'], o['torque'],
                                o['emg_std_long'], o['emg_mean_long'], opt['calu_torque'])
        elif joint_idx == 'waist':
            for i in range(len(idx)):
                print('-' * 25, i, '-' * 25)
                o = read_realted_files_dl(label, idx[i])
                t1, r, calu_torque1, y_r = \
                    optimization_pyomo(o['emg_mean'], o['emg_std'], o['arm1'], o['torque1'], o['time'], o['emg'])
                plot_all_result(1, t1, r, o['emg'], o['time'], o['torque1'], o['emg_std_long'],
                                o['emg_mean_long'], calu_torque1)
        elif joint_idx == 'hip':
            for i in range(len(idx)):
                print('-' * 25, i, '-' * 25)
                o = read_realted_files_dl(label, idx[i])
                t1, r, calu_torque1, y_r = \
                    optimization_pyomo(o['emg_mean'], o['emg_std'], o['arm2'], o['torque2'], o['time'], o['emg'].T)
                plot_all_result(1, t1, r, o['emg'], o['time'], o['torque2'], o['emg_std_long'],
                                o['emg_mean_long'], calu_torque1)
        elif joint_idx == 'knee':
            for i in range(len(idx)):
                print('-' * 25, i, '-' * 25)
                o = read_realted_files_dl(label, idx[i])
                t1, r, calu_torque1, y_r = \
                    optimization_pyomo(o['emg_mean'], o['emg_std'], o['arm3'], o['torque3'], o['time'], o['emg'].T)
                plot_all_result(1, t1, r, o['emg'], o['time'], o['torque3'], o['emg_std_long'],
                                o['emg_mean_long'], calu_torque1)


def plot_mat(num, length, t_all, r_all, emg_all, time_all, torque_all):
    # plt.figure(figsize=(9, 7.8))
    plt.figure(figsize=(6, 7.5))
    mus_idx = ['BIC', 'BRA', 'BRD', 'TRI']
    for i in range(num):
        # t = t_all[i * length:(i + 1) * length]
        # r = r_all[:, i * length:(i + 1) * length]
        # emg = emg_all[:, i * length:(i + 1) * length]
        # time = time_all[i * length:(i + 1) * length]
        # torque = torque_all[i * length:(i + 1) * length]
        t = t_all
        r = r_all
        emg = emg_all
        time = time_all
        torque = torque_all

        # plt.subplot(len(mus_idx) + 1, num, i + 1)
        # plt.plot(time, np.asarray(t), label='optimization', linewidth=2)
        # plt.plot(time, torque, label='measured', linewidth=2)
        plt.subplot(len(mus_idx) + 1, num, i + 1)
        plt.plot(np.asarray(t), label='optimization', linewidth=2)
        plt.plot(torque, label='measured', linewidth=2)
        plt.axvline(x=length, color='darkred', linestyle='--')
        plt.axvline(x=length * 2, color='darkred', linestyle='--')
        plt.gca().get_xaxis().set_ticklabels([])

        if i == 0:
            plt.ylabel('torque', weight='bold', size=10)
        plt.legend()
        # rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
        # print("torque rmse", idx, ":\t", "{:.2f}".format(rmse))

        for j in range(len(mus_idx)):
            plt.subplot(len(mus_idx) + 1, num, (j + 1) * num + i + 1)
            # if plot_distribution is True:
            #     if mvc_is_variable is True:
            #         plt.errorbar(time, emg_mean_long[j, :] * c[j], 2 * emg_std_long[j, :] * c[j], label='emg',
            #                      color='lavender', zorder=1)
            #     else:
            #         plt.errorbar(time, emg_mean_long[j, :], 2 * emg_std_long[j, :], label='emg', color='lavender',
            #                      zorder=1)
            # else:
            #     plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
            if j == len(mus_idx) - 1:
                # plt.plot(time, np.asarray(emg[j + 1]), label='emg', linewidth=2, zorder=3)
                # plt.plot(time, np.asarray(r[j + 1, :]), label='optimization', linewidth=2, zorder=2)
                # plt.xlabel('time (s)', weight='bold', size=10)
                plt.plot(np.asarray(r[j + 1, :]), label='optimization', linewidth=2, zorder=2)
                plt.plot(np.asarray(emg[j + 1]), label='emg', linewidth=2, zorder=3)
                plt.axvline(x=length, color='darkred', linestyle='--')
                plt.axvline(x=length * 2, color='darkred', linestyle='--')
                plt.xlabel('timestep', weight='bold', size=10)
                plt.legend()
            else:
                # plt.plot(time, np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                # plt.plot(time, np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
                plt.plot(np.asarray(r[j, :]), label='optimization', linewidth=2, zorder=2)
                plt.plot(np.asarray(emg[j]), label='emg', linewidth=2, zorder=3)
                plt.axvline(x=length, color='darkred', linestyle='--')
                plt.axvline(x=length * 2, color='darkred', linestyle='--')
                plt.gca().get_xaxis().set_ticklabels([])
                plt.legend()
            if i == 0:
                plt.ylabel(mus_idx[j], weight='bold')
            # plt.tight_layout()
            # plt.legend()


if __name__ == '__main__':
    # calculate_other_emg_distribution(label='yt-bp-all')
    # # calculate_chenzui_emg_distribution(label='bp-4kg')
    # # calculate_lizhuo_emg_distribution(label='bp-4kg')
    # # plt.show()
    # #
    # # # label = 'chenzui-left-all-4kg-cts'
    # # # idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    # # # label = 'chenzui-left-all-5.5kg-cts'
    # # # idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    # # label = 'bp-chenzui-left-5.5kg'
    # # idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    label = 'bp-yuetian-right-40kg'
    idx = ['1', '2', '3', '4', '5']
    # one_repetition(label, idx)
    # plt.show()

    o = read_groups_files(label, idx)
    if 'bp' in label:
        if joint_idx == 'all':
            opt = optimization_pyomo_deadlift(o['emg_mean'], o['emg_std'], o['arm'], o['torque'], o['emg'])
            print('-' * 25, 'training_rmse', '-' * 25)
            plot_all_result(len(idx), opt['torque'], opt['activation'], o['emg'].T, o['time'], o['torque'],
                            o['emg_std_long'], o['emg_mean_long'], opt['calu_torque'])
            np.save('fmax', opt['fmax'])
            # y_r = np.load('fmax.npy')
            print('-' * 25, 'application', '-' * 25)
            application(label, opt['fmax'], idx)
    elif 'dl' in label:
        if joint_idx == 'all':
            opt = optimization_pyomo_deadlift(o['emg_mean'], o['emg_std'], o['arm'], o['torque'], o['emg'])
            print('-' * 25, 'training_rmse', '-' * 25)
            plot_all_result(len(idx), opt['torque'], opt['activation'], o['emg'].T, o['time'], o['torque'],
                            o['emg_std_long'], o['emg_mean_long'], opt['calu_torque'])
            np.save('fmax', opt['fmax'])
            # y_r = np.load('fmax.npy')
            print('-' * 25, 'application', '-' * 25)
            application(label, opt['fmax'], idx)
        elif joint_idx == 'twist' or joint_idx == 'hip' or joint_idx == 'knee':
            t1, r, calu_torque1, y_r = \
                optimization_pyomo(o['emg_mean'], o['emg_std'], o['arm' + joint_num[joint_idx]],
                                   o['torque' + joint_num[joint_idx]], o['time'], o['emg'])
            plot_all_result(len(idx), t1, r, o['emg'].T, o['time'], o['torque' + joint_num[joint_idx]],
                            o['emg_std_long'], o['emg_mean_long'], calu_torque1)
    #
    # print('-' * 25, 'training rmse', '-' * 25)
    # c = 0
    # # plot_all_result(1, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    # plot_all_result_bp(9, t1, t2, r, files['emg'], files['time'], files['torque1'], files['torque2'],
    #                    files['emg_std_long'], files['emg_mean_long'], calu_torque1, calu_torque2, c)
    #
    # # y_r = np.asarray([8547.72, 17095.45, 1709.54])
    # # y_r = np.asarray([11123.34, 11123.34, 1112.33])
    # # y_r = np.asarray([10744.52, 10744.52, 1074.45])
    # # y_r = np.asarray([7592.42, 15184.84, 1670.72])
    # # y_r = np.asarray([4176.24, 28956.69, 788.10])
    # # y_r = np.asarray([86.50, 2679.23, 64.00, 143.01, 4985.50, 4849.45])
    # # y_r = np.asarray([86.50, 111.50, 64.00, 57199.92, 5853.39, 5520.00,
    # #                   86.50, 86.50, 86.50, 86.50, 86.50, 86.50, 86.50, 86.50, 86.50, 86.50, 86.50,
    # #                   111.50, 111.50, 111.50,
    # #                   64.00, 64.00, 64.00,
    # #                   57199.92, 57199.92, 57199.92, 57199.92, 57199.92, 57199.92, 57199.92,
    # #                   4985.50, 4985.50, 4985.50, 4985.50, 4985.50,
    # #                   4849.45, 4849.45, 4849.45, 4849.45, 4849.45])
    # print('-' * 25, 'application', '-' * 25)
    # # np.save('fmax', y_r)
    # y_r = np.load('fmax.npy')
    # print(y_r)
    # application('bp-chenzui-left-4kg', y_r)
    # application('bp-chenzui-left-5.5kg', y_r)
    # application('bp-chenzui-left-6.5kg', y_r)
    # application('bp-chenzui-left-7kg', y_r)
    # application('bp-chenzui-left-7kg', y_r)
    # application('bp-yuetian-right-50kg', y_r)

    # application('chenzui-left-all-4kg', y_r)
    # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    #     = read_realted_files(label='chenzui-left-all-2.5kg', idx='8')
    # test_optimization_emg(3, y_r, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std,
    #                       emg_trend_u, emg_trend_d)
    # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    #     = read_realted_files(label='chenzui-left-all-6.5kg', idx='18')
    # test_optimization_emg(4, y_r, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std,
    #                       emg_trend_u, emg_trend_d)

    plt.show()
